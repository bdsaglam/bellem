import json
import os
from math import ceil

from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.lang.llama import prepare_llama2_for_training
from bellek.ml.transformers import load_tokenizer_model
from bellek.utils import NestedDict, flatten_dict, generate_time_id

log = get_logger(__name__)


def prepare_config(config: NestedDict):
    from bellek.ml.transformers import preprocess_config as tpc

    # Setup config related to transformers library
    config = tpc(config)

    # Generate unique model id
    model_id = config.at("hfhub.model_id")
    model_id += "-peft"
    if "debug" not in model_id:
        model_id += f"-{generate_time_id()}"
    config.set("hfhub.model_id", model_id)

    return config


def before_experiment(wandb_run):
    config = prepare_config(NestedDict.from_flat_dict(wandb_run.config))
    config.set("wandb.run_id", wandb_run.id)
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)

    # Wandb env variables
    os.environ["WANDB_PROJECT"] = wandb_run.project
    os.environ["WANDB_LOG_MODEL"] = "end"

    # Set random seed
    if seed := config.get("seed"):
        from fastai.torch_core import set_seed

        set_seed(seed)

    # Save preprocessed config
    with open("./config.proc.json", "w") as f:
        json.dump(config, f, indent=2)


def train(wandb_run, config: NestedDict):
    # Base model
    pretrained_model_config = config["pretrained_model"]
    model_id = pretrained_model_config.pop("model_name_or_path")
    log.info(f"Loading base model {model_id}")
    tokenizer, base_model = load_tokenizer_model(model_id, **pretrained_model_config)
    prepare_llama2_for_training(tokenizer, base_model)

    # Train dataset
    train_ds = load_dataset(**config.at("dataset.train"))
    log.info(f"Loaded training dataset with {len(train_ds)} samples.")

    # Inspect token counts
    tokenized_train_ds = train_ds.map(lambda examples: tokenizer(examples["text"]), batched=True)
    token_counts = [len(input_ids) for input_ids in tokenized_train_ds["input_ids"]]
    log.info(f"Input token counts: min={min(token_counts)}, max={max(token_counts)}")

    # Supervised fine-tuning
    if config.at("trainer.max_seq_length") is None:
        config.set("trainer.max_seq_length", ceil(max(token_counts) / 8) * 8)
    max_seq_length = config.at("trainer.max_seq_length")
    log.info(f"Setting max_seq_length={max_seq_length}")
    packing = config.at("trainer.packing", False)
    if response_template := config.at("trainer.response_template"):
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    else:
        data_collator = None
    peft_config = LoraConfig(**config.at("trainer.lora", {}))
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

    return trainer


def after_experiment(wandb_run, config):
    with open("./config.after.json", "w") as f:
        json.dump(config, f, indent=2)


def run_experiment(wandb_run):
    before_experiment(wandb_run)
    config = NestedDict.from_flat_dict(wandb_run.config)
    train(wandb_run, config)
    after_experiment(wandb_run, config)


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
