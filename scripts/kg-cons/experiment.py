import os
from typing import Any, Dict, List, Tuple  # noqa: F401

import evaluate
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import wandb
from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.ml.kg.cons import parse_triplet_strings
from bellek.utils import NestedDict

DEVICE_MAP = {"": 0}


log = get_logger()


def get_default_bnb_config(
    # Activate 4-bit precision base model loading
    use_4bit=True,
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype="float16",
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type="nf4",
    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant=False,
):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            log.warning("Your GPU supports bfloat16: accelerate training with bf16=True")

    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )


def load_model_tokenizer(
    model_name_or_path,
    bnb_config=None,
    device_map=None,
):
    if bnb_config is None:
        bnb_config = get_default_bnb_config()

    # Load the entire model on the GPU 0
    if device_map is None:
        device_map = DEVICE_MAP

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 trainin
    return model, tokenizer


def prepare_erx_dataset(split=None):
    from bellek.ml.kg.dataset import batch_transform_webnlg

    ds = load_dataset("web_nlg", "release_v3.0_en", split=split)
    column_names = list(ds.column_names.values())[0] if isinstance(ds, DatasetDict) else ds.column_names
    return ds.map(batch_transform_webnlg, batched=True, remove_columns=column_names)


def make_erx_formatter(erx_dataset):
    from bellek.ml.kg.cons import ERXFormatter

    few_shot_examples = erx_dataset.select(list(range(3)))
    return ERXFormatter(few_shot_examples=few_shot_examples)


def prepare_training_dataset(ds, erx_formatter):
    return ds.map(erx_formatter.format_for_train)


def prepare_evaluation_dataset(ds, erx_formatter):
    return ds.map(erx_formatter.format_for_inference)


def evaluate_finetuned_model(wandb_run, tokenizer, model, evaluation_dataset):
    from transformers import pipeline

    config = NestedDict.from_flat_dict(wandb_run.config)
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.at("evaluation.max_new_tokens", 128),
        batch_size=config.at("evaluation.batch_size", 4),
        return_full_text=False,
        device=0,
    )

    def _evaluate(pipe, dataset):
        metric = evaluate.load("bdsaglam/jer")
        results = pipe(dataset["text"])
        generated_texts = [result[0]["generated_text"] for result in results]
        predictions = [parse_triplet_strings(gen_text.strip()) for gen_text in generated_texts]
        references = dataset["triplets"]
        scores = metric.compute(predictions=predictions, references=references)
        return generated_texts, predictions, scores

    generated_texts, predicted_triplets, scores = _evaluate(pipe, evaluation_dataset)
    evaluation_df = evaluation_dataset.to_pandas()
    evaluation_df["generated_texts"] = generated_texts
    evaluation_df["predicted_triplets"] = predicted_triplets
    wandb_run.log(
        {
            **scores,
            "evaluation-dataframe": wandb.Table(dataframe=evaluation_df.reset_index()),
        }
    )


def run_experiment(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)
    if seed := config.get("seed"):
        from fastai.torch_core import set_seed

        set_seed(seed)

    os.environ["WANDB_PROJECT"] = wandb_run.project
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # Base model
    model_id = config.at("base_model_id")
    log.info(f"Loading base model {model_id}")
    base_model, tokenizer = load_model_tokenizer(model_id)

    # Dataset
    log.info("Preparing entity-extraction dataset")
    train_erx_ds = prepare_erx_dataset(split=config.at("dataset.train.split"))
    rel_ds = train_erx_ds.map(
        lambda example: dict(relations=[triplet.split("|")[1].strip() for triplet in example["triplets"]])
    )
    relation_set = {rel for rels in rel_ds["relations"] for rel in rels}
    log.info(f"Number of unique relations:{len(relation_set)}")
    log.info(f"Number of tokens for all relations: {len(tokenizer.encode(' '.join(relation_set)))}")
    erx_formatter = make_erx_formatter(train_erx_ds)

    # Instruction tuning dataset
    log.info("Preparing training dataset")
    train_ds = prepare_training_dataset(train_erx_ds, erx_formatter)
    tokenized_datasets = train_ds.map(lambda examples: tokenizer(examples["text"]), batched=True)
    token_counts = [len(input_ids) for input_ids in tokenized_datasets["input_ids"]]
    log.info(f"Input token counts: min={min(token_counts)}, max={max(token_counts)}")

    ## PEFT
    peft_config = LoraConfig(**config.at("fine_tuning.lora", {}))

    # Supervised fine-tuning
    training_args = TrainingArguments(
        output_dir="./results",
        **config["training"],
    )
    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=None,
        packing=False,
        args=training_args,
    )
    log.info("Training model")
    trainer.train()

    # Save trained model
    log.info("Saving model")
    final_model_id = config.at("hfhub.model_id")
    trainer.model.save_pretrained(final_model_id.split("/", 1)[-1])
    trainer.model.push_to_hub(final_model_id)
    log.info(f"Uploaded PEFT adapters to HF Hub as {final_model_id}")

    # Evaluate model
    log.info("Evaluating model")
    eval_erx_ds = prepare_erx_dataset(split=config.at("dataset.eval.split"))
    eval_ds = prepare_evaluation_dataset(eval_erx_ds, erx_formatter)
    evaluate_finetuned_model(wandb_run, tokenizer, trainer.model, eval_ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(run_experiment, args)
