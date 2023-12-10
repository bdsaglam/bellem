import os
from copy import deepcopy
from time import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import wandb

from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.ml.kg.cons import parse_triplet_strings
from bellek.ml.llama import prepare_llama2_for_inference, prepare_llama2_for_training
from bellek.ml.transformers import load_tokenizer_model
from bellek.utils import NestedDict, flatten_dict

log = get_logger(__name__)


def preprocess_config(config: NestedDict):
    config = deepcopy(config)

    # Set float precision
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        log.info("GPU supports bfloat16.")
        torch_dtype, bf16, fp16, bnb_4bit_compute_dtype = ("bfloat16", True, False, "bfloat16")
    else:
        log.info("GPU does not support bfloat16.")
        torch_dtype, bf16, fp16, bnb_4bit_compute_dtype = ("float16", False, True, "float16")

    if config.at("pretrained_model.torch_dtype"):
        config.set("pretrained_model.torch_dtype", torch_dtype)
    if config.at("pretrained_model.quantization_config.load_in_4bit"):
        config.set("pretrained_model.quantization_config.bnb_4bit_compute_dtype", bnb_4bit_compute_dtype)
    if config.at("trainer.training_args.bf16") or config.at("trainer.training_args.fp16"):
        config.set("trainer.training_args.bf16", bf16)
        config.set("trainer.training_args.fp16", fp16)

    # Update model id
    timestamp = int(time())
    model_id = config.at("hfhub.model_id")
    config.set("hfhub.model_id", f"{model_id}-{timestamp}")

    return config


def load_ds(dataset_config):
    return load_dataset(
        dataset_config["path"],
        dataset_config.get("name"),
        split=dataset_config.get("split"),
    )


def evaluate_finetuned_model(wandb_run, tokenizer, model, evaluation_dataset):
    from transformers import pipeline

    config = NestedDict.from_flat_dict(wandb_run.config)

    # prepare evaluation dataset
    def extract_triplets(example):
        response_template = config.at("trainer.response_template")
        assert response_template is not None
        output = example["text"].rsplit(response_template, 1)[-1].strip()
        triplet_strings = parse_triplet_strings(output)
        return {"triplet_strings": triplet_strings}

    evaluation_dataset = evaluation_dataset.map(extract_triplets)

    # setup generation pipeline
    prepare_llama2_for_inference(tokenizer, model)
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
        references = dataset["triplet_strings"]
        scores = metric.compute(predictions=predictions, references=references)
        return generated_texts, predictions, scores

    generated_texts, predicted_triplets, scores = _evaluate(pipe, evaluation_dataset)

    # log predictions to wandb
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
    pretrained_model_config = config["pretrained_model"]
    model_id = pretrained_model_config.pop("model_name_or_path")
    log.info(f"Loading base model {model_id}")
    tokenizer, base_model = load_tokenizer_model(model_id, **pretrained_model_config)
    prepare_llama2_for_training(tokenizer, base_model)

    # Train dataset
    log.info("Preparing training dataset")
    train_ds = load_ds(config.at("dataset.train"))
    tokenized_datasets = train_ds.map(lambda examples: tokenizer(examples["text"]), batched=True)
    token_counts = [len(input_ids) for input_ids in tokenized_datasets["input_ids"]]
    log.info(f"Input token counts: min={min(token_counts)}, max={max(token_counts)}")

    # Supervised fine-tuning
    peft_config = LoraConfig(**config.at("trainer.lora", {}))
    max_seq_length = config.at("trainer.max_seq_length", max(token_counts))
    packing = config.at("trainer.packing", False)
    if response_template := config.at("trainer.response_template"):
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    else:
        data_collator = None
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
        data_collator=data_collator,
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

    # Merge adapters to model and publish
    log.info("Merging adapters to model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model_id = f"{final_model_id}-merged"
    model.push_to_hub(merged_model_id)
    tokenizer.push_to_hub(merged_model_id)
    log.info(f"Uploaded merged model to HF Hub as {merged_model_id}")

    # Evaluate model
    if val_ds_config := config.at("dataset.validation"):
        val_ds = load_ds(val_ds_config)
        log.info(f"Evaluating model on validation dataset with {len(val_ds)} samples.")
        evaluate_finetuned_model(wandb_run, tokenizer, trainer.model, val_ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    prepare_config_kwargs = {
        "exclude_resolving_paths": [
            "pretrained_model.model_name_or_path",
            "dataset.train.path",
            "dataset.validation.path",
        ]
    }
    main(run_experiment, args, prepare_config_kwargs)
