import os
from typing import Any, Dict, List, Tuple  # noqa: F401

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from bellek.ml.experiment import main
from bellek.utils import NestedDict

DEVICE_MAP = {"": 0}


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
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

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


def prepare_erx_dataset(split = None):
    from bellek.ml.kg.dataset import batch_transform_webnlg
    ds = load_dataset("web_nlg", "release_v3.0_en", split=split)
    column_names = list(ds.column_names.values())[0] if isinstance(ds, DatasetDict) else ds.column_names
    return ds.map(batch_transform_webnlg, batched=True, remove_columns=column_names)


def prepare_finetuning_dataset(erx_dataset):
    from bellek.ml.kg.cons import ERXFormatter

    few_shot_examples = erx_dataset.select(list(range(3)))
    formatter = ERXFormatter(few_shot_examples=few_shot_examples)
    return erx_dataset.map(formatter.format_for_train)


def run_experiment(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)
    if seed := config.get("seed"):
        from fastai.torch_core import set_seed

        set_seed(seed)

    os.environ["WANDB_PROJECT"] = wandb_run.project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # Base model
    model_name = config.at("base_model.name")
    print(f"Loading base model {model_name}")
    base_model, tokenizer = load_model_tokenizer(model_name)

    # Dataset
    print("Preparing entity-extraction dataset")
    erx_ds = prepare_erx_dataset(subset=config.at("dataset.split"))
    rel_ds = erx_ds.map(
        lambda example: dict(relations=[triplet.split("|")[1].strip() for triplet in example["triplets"]])
    )
    relation_set = {rel for rels in rel_ds["relations"] for rel in rels}
    print("Number of unique relations:", len(relation_set))
    print(
        "Number of tokens for all relations:",
        len(tokenizer.encode(" ".join(relation_set))),
    )

    # Fine-tuning
    print("Preparing fine-tuning dataset")
    finetune_ds = prepare_finetuning_dataset(erx_ds)
    tokenized_datasets = finetune_ds.map(lambda examples: tokenizer(examples["text"]), batched=True)
    token_counts = [len(input_ids) for input_ids in tokenized_datasets["input_ids"]]
    print(f"Input token counts: min={min(token_counts)}, max={max(token_counts)}")

    ## PEFT
    peft_config = LoraConfig(**config.at("fine_tuning.lora", {}))

    # Supervised fine-tuning
    training_args = TrainingArguments(
        output_dir="./results",
        **config["train"],
    )
    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=finetune_ds,
        dataset_text_field="text",
        max_seq_length=None,
        packing=False,
        args=training_args,
    )

    # Train model
    print("Training model")
    trainer.train()

    # Save trained model
    final_model_name = f"{model_name.split('/')[-1]}-kg-cons"
    trainer.model.save_pretrained(final_model_name)
    trainer.model.push_to_hub(final_model_name)
    print(f"Uploaded PEFT adapters to HF Hub with name {final_model_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(run_experiment, args)
