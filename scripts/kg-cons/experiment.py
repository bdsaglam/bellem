import json
import os
from math import ceil
from time import time

from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import wandb
from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.ml.kg.cons import parse_triplet_strings
from bellek.ml.llama import prepare_llama2_for_inference, prepare_llama2_for_training
from bellek.ml.transformers import load_tokenizer_model
from bellek.utils import NestedDict, flatten_dict

log = get_logger(__name__)


def prepare_config(config: NestedDict):
    from bellek.ml.transformers import preprocess_config as tpc

    # Setup config related to transformers library
    config = tpc(config)

    # Generate unique model id
    timestamp = int(time())
    model_id = config.at("hfhub.model_id")
    config.set("hfhub.model_id", f"{model_id}-peft-{timestamp}")

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


def load_ds(dataset_config):
    return load_dataset(
        dataset_config["path"],
        dataset_config.get("name"),
        split=dataset_config.get("split"),
    )


def train(wandb_run, config: NestedDict):
    # Base model
    pretrained_model_config = config["pretrained_model"]
    model_id = pretrained_model_config.pop("model_name_or_path")
    log.info(f"Loading base model {model_id}")
    tokenizer, base_model = load_tokenizer_model(model_id, **pretrained_model_config)
    prepare_llama2_for_training(tokenizer, base_model)

    # Train dataset
    log.info("Preparing training dataset")
    train_ds = load_ds(config.at("dataset.train"))

    # Inspect token counts
    tokenized_train_ds = train_ds.map(lambda examples: tokenizer(examples["text"]), batched=True)
    token_counts = [len(input_ids) for input_ids in tokenized_train_ds["input_ids"]]
    log.info(f"Input token counts: min={min(token_counts)}, max={max(token_counts)}")

    # Supervised fine-tuning
    peft_config = LoraConfig(**config.at("trainer.lora", {}))
    max_seq_length = config.at("trainer.max_seq_length", ceil(max(token_counts) / 8) * 8)
    log.info(f"Setting max_seq_length={max_seq_length}")
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

    return trainer


def evaluate_finetuned_model(wandb_run, config, tokenizer, model):
    from transformers import pipeline

    val_ds_config = config.at("dataset.validation")
    if val_ds_config is None:
        return

    val_ds = load_ds(val_ds_config)
    log.info(f"Evaluating model on validation dataset with {len(val_ds)} samples.")

    # prepare evaluation dataset
    def extract_triplets(example):
        response_template = config.at("trainer.response_template")
        assert response_template is not None
        output = example["text"].rsplit(response_template, 1)[-1]
        return {"output": output}

    val_ds = val_ds.map(extract_triplets)

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
        generations = [result[0]["generated_text"] for result in results]
        predictions = [parse_triplet_strings(text.strip()) for text in generations]
        references = [parse_triplet_strings(text.strip()) for text in dataset["output"]]
        scores = metric.compute(predictions=predictions, references=references)
        return generations, predictions, references, scores

    generations, predictions, references, scores = _evaluate(pipe, val_ds)

    # log predictions to wandb
    evaluation_df = val_ds.to_pandas()
    evaluation_df["generation"] = generations
    evaluation_df["prediction"] = predictions
    evaluation_df["reference"] = references
    wandb_run.log(
        {
            **scores,
            "evaluation-dataframe": wandb.Table(dataframe=evaluation_df.reset_index()),
        }
    )


def after_experiment(wandb_run, config):
    with open("./config.after.json", "w") as f:
        json.dump(config, f, indent=2)


def run_experiment(wandb_run):
    before_experiment(wandb_run)
    config = NestedDict.from_flat_dict(wandb_run.config)
    trainer = train(wandb_run, config)
    evaluate_finetuned_model(wandb_run, config, trainer.tokenizer, trainer.model)
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
