import os
from copy import deepcopy

import torch
from datasets import DatasetDict, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
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
from bellek.utils import NestedDict, flatten_dict

log = get_logger(__name__)


def preprocess_config(config: NestedDict):
    config = deepcopy(config)

    # Use bfloat16 if GPU supports
    if (
        config.at("trainer.training_args.bf16")
        or config.at("trainer.training_args.fp16")
        or config.at("pretrained_model.quantization_config.load_in_4bit")
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            log.info("GPU supports bf16.")
            bf16, fp16, bnb_4bit_compute_dtype = (True, False, "bfloat16")
        else:
            log.info("GPU does not support bf16, using fp16.")
            bf16, fp16, bnb_4bit_compute_dtype = (False, True, "float16")
        config.set("trainer.training_args.bf16", bf16)
        config.set("trainer.training_args.fp16", fp16)
        if config.at("pretrained_model.quantization_config.load_in_4bit"):
            config.set("pretrained_model.quantization_config.bnb_4bit_compute_dtype", bnb_4bit_compute_dtype)
    return config


def load_model_tokenizer(
    model_name_or_path: str,
    *,
    auto_model_cls=AutoModelForCausalLM,
    device_map={"": 0},
    **model_kwargs,
):
    # Setup quantization config
    if (quantization_config := model_kwargs.get("quantization_config")) and isinstance(quantization_config, dict):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_config)
    # Load model
    model = auto_model_cls.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.float16,
        **model_kwargs,
    )
    # Load tokenizer
    tokenizer_id = (
        model.active_peft_config.base_model_name_or_path
        if auto_model_cls == AutoPeftModelForCausalLM
        else model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    return tokenizer, model


def fix_llama_model_tokenizer(tokenizer, model):
    # fix tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    # fix model
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return tokenizer, model


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
    )

    def _evaluate(pipe, dataset):
        import evaluate

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
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)

    if seed := config.get("seed"):
        from fastai.torch_core import set_seed

        set_seed(seed)

    os.environ["WANDB_PROJECT"] = wandb_run.project
    os.environ["WANDB_LOG_MODEL"] = "end"

    # Base model
    model_id = config.at("base_model_id")
    log.info(f"Loading base model {model_id}")
    tokenizer, base_model = fix_llama_model_tokenizer(
        *load_model_tokenizer(
            model_id,
            **config.at("pretrained_model", {}),
        )
    )

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

    # Supervised fine-tuning
    peft_config = LoraConfig(**config.at("trainer.lora", {}))
    max_seq_length = config.at("trainer.max_seq_length")
    packing = config.at("trainer.packing", False)
    training_args = TrainingArguments(
        output_dir="./results",
        **config.at("trainer.training_args"),
    )
    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=packing,
        args=training_args,
    )
    log.info("Training model")
    trainer.train()

    # Save trained model
    log.info("Saving model")
    final_model_id = config.at("hfhub.model_id")
    trainer.model.push_to_hub(final_model_id)
    tokenizer.push_to_hub(final_model_id)
    log.info(f"Uploaded PEFT adapters to HF Hub as {final_model_id}")

    # Evaluate model
    if eval_split := config.at("dataset.eval.split"):
        log.info("Evaluating model")
        eval_erx_ds = prepare_erx_dataset(split=eval_split)
        eval_ds = prepare_evaluation_dataset(eval_erx_ds, erx_formatter)
        evaluate_finetuned_model(wandb_run, tokenizer, trainer.model, eval_ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(run_experiment, args)
