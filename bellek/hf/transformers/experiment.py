# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/hf.transformers.experiment.ipynb.

# %% auto 0
__all__ = ['log', 'preprocess_config', 'fine_tune', 'make_io_dataset', 'make_pipeline', 'flat_pipeline', 'evaluate_pipeline',
           'evalu8']

# %% ../../../nbs/hf.transformers.experiment.ipynb 3
from copy import deepcopy
from typing import Any, Callable

import torch
from datasets import Dataset, load_dataset
from transformers import pipeline

from .utils import load_tokenizer_model
from ...logging import get_logger
from ...utils import NestedDict, generate_time_id

log = get_logger(__name__)

# %% ../../../nbs/hf.transformers.experiment.ipynb 4
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
    
    # Generate unique model id
    model_id = config.at("hfhub.model_id")
    if config.at("trainer.lora") and "-peft" not in model_id:
        model_id += "-peft"
    if "debug" not in model_id:
        model_id += f"-{generate_time_id()}"
    config.set("hfhub.model_id", model_id)

    return config


# %% ../../../nbs/hf.transformers.experiment.ipynb 5
def fine_tune(config: NestedDict):
    from math import ceil
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    # Base model
    pretrained_model_config = config["pretrained_model"]
    model_id = pretrained_model_config.pop("model_name_or_path")
    tokenizer, base_model = load_tokenizer_model(model_id, **pretrained_model_config)
    log.info(f"Loaded base model {model_id}")

    if "llama" in model_id:
        from bellek.hf.transformers.llama import prepare_llama2_for_training

        log.info("Base model is a LLAMA model, preparing it for training.")
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
        log.info(f"Using completion-only data collator with response template '{response_template}'")
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
    log.info("Starting training...")
    trainer.train()

    # Save trained model
    log.info("Saving model...")
    final_model_id = config.at("hfhub.model_id")
    trainer.model.push_to_hub(final_model_id)
    tokenizer.push_to_hub(final_model_id)
    log.info(f"Uploaded PEFT adapters to HF Hub as {final_model_id}")

    return trainer


# %% ../../../nbs/hf.transformers.experiment.ipynb 6
def make_io_dataset(dataset: Dataset, response_template: str) -> Dataset:
    def extract_input_output(example):
        input, output = example["text"].rsplit(response_template, 1)
        input += response_template
        return {"input": input, "output": output}

    return dataset.map(extract_input_output)


def _load_tokenizer_model(config: NestedDict):
    model_id = config.at("hfhub.model_id")
    quantization_config = config.at("pretrained_model.quantization_config")
    return load_tokenizer_model(
        model_id,
        quantization_config=quantization_config,
    )

def make_pipeline(config, tokenizer, model):
    model_id = config.at("hfhub.model_id")
    if "llama" in model_id:
        from bellek.hf.transformers.llama import prepare_llama2_for_inference

        prepare_llama2_for_inference(tokenizer, model)

    # Create pipeline
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **config.at("evaluation.pipeline", {}),
    )


def flat_pipeline(pipe):
    def func(inputs) -> list[str]:
        results = pipe(inputs)
        return [result[0]["generated_text"] for result in results]

    return func


def evaluate_pipeline(
    dataset,
    pipe,
    *,
    metric,
    output_parse_fn: Callable[[str], Any] | None = None,
):
    eos_token = pipe.tokenizer.special_tokens_map["eos_token"]
    def parse_output(text):
        text = text.replace(eos_token, "").strip()
        if output_parse_fn:
            text = output_parse_fn(text)
        return text

    log.info(f"Running pipeline on dataset with {len(dataset)} samples...")
    generations = flat_pipeline(pipe)(dataset["input"])

    predictions = [parse_output(text) for text in generations]
    references = [parse_output(text) for text in dataset["output"]]

    dataf = dataset.to_pandas()
    dataf["generation"] = generations
    dataf["prediction"] = predictions
    dataf["reference"] = references

    scores = metric.compute(predictions=predictions, references=references)

    return scores, dataf


def evalu8(
    config,
    *,
    tokenizer=None,
    model=None,
    output_parse_fn: Callable[[str], Any] | None = None,
):
    import evaluate

    # Load validation dataset
    ds_config = config.at("dataset.validation")
    assert ds_config
    ds = load_dataset(**ds_config)
    assert len(ds) > 0, "Dataset is empty!"

    # Ensure the dataset has input/output columns
    cols = ds[0].keys()
    if "input" not in cols or "output" not in cols:
        response_template = config.at("trainer.response_template")
        assert response_template
        ds = make_io_dataset(ds, response_template)

    # Prepare text generation pipeline
    if tokenizer is None or model is None:
        tokenizer, model = _load_tokenizer_model(config)
    pipe = make_pipeline(config, tokenizer, model)

    # Load evaluation metric
    metric = evaluate.load(config.at("evaluation.metric"))

    return evaluate_pipeline(
        ds,
        pipe,
        metric=metric,
        output_parse_fn=output_parse_fn,
    )
