# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/hf.transformers.experiment.ipynb.

# %% auto 0
__all__ = ['log', 'prepare_config_for_fp', 'preprocess_config', 'make_datacollator', 'prepare_model_for_training',
           'calculate_token_counts', 'fine_tune', 'prepare_model_for_inference', 'make_pipeline', 'predict',
           'evaluate_']

# %% ../../../nbs/hf.transformers.experiment.ipynb 3
from copy import deepcopy
from math import ceil
from typing import Any, Callable

import torch
from datasets import Dataset
from transformers import TrainingArguments, pipeline
from transformers.pipelines.text_generation import Chat
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from .generation import generate
from .utils import load_tokenizer_model
from ..datasets.utils import load_datasets
from ...lang.dataset import partition_input_output_messages
from ...logging import get_logger
from ...utils import generate_time_id
from ...ds import NestedDict
from ...torch.dataset.utils import ListDataset

log = get_logger(__name__)

# %% ../../../nbs/hf.transformers.experiment.ipynb 4
def prepare_config_for_fp(config: NestedDict):
    if not torch.cuda.is_available():
        return config

    # Set float precision
    if config.at("pretrained_model.torch_dtype") in {"float16", "bfloat16"}:
        major, _ = torch.cuda.get_device_capability()
        gpu_supports_bf = major >= 8
        if gpu_supports_bf:
            log.info("GPU supports bfloat16.")
        else:
            log.info("GPU does not support bfloat16.")
        
        if config.at("pretrained_model.torch_dtype") == "bfloat16" and gpu_supports_bf:
            log.info("Using bfloat16.")
            torch_dtype, bf16, fp16, bnb_4bit_compute_dtype = ("bfloat16", True, False, "bfloat16")
        else:
            log.info("Using float16.")
            torch_dtype, bf16, fp16, bnb_4bit_compute_dtype = ("float16", False, True, "float16")
    else:
        log.info("Not using half-precision float.")
        torch_dtype, bf16, fp16, bnb_4bit_compute_dtype = (None, None, None, None)

    if config.at("pretrained_model.torch_dtype"):
        config.set("pretrained_model.torch_dtype", torch_dtype)
    if config.at("pretrained_model.quantization_config.load_in_4bit"):
        config.set("pretrained_model.quantization_config.bnb_4bit_compute_dtype", bnb_4bit_compute_dtype)
    if config.at("trainer.training_args.bf16") or config.at("trainer.training_args.fp16"):
        config.set("trainer.training_args.bf16", bf16)
        config.set("trainer.training_args.fp16", fp16)
    return config

def preprocess_config(config: NestedDict):
    config = deepcopy(config)

    if isinstance(config.at("dataset.train"), dict):
        config.set("dataset.train", [config.at("dataset.train")])
    if isinstance(config.at("dataset.validation"), dict):
        config.set("dataset.validation", [config.at("dataset.validation")])
    
    if config.at("distributed_training"):
        from accelerate import PartialState
        config.set("pretrained_model.device_map", {"": PartialState().process_index})

    config = prepare_config_for_fp(config)
    
    # Generate unique model id for output model
    if (out_model_id := config.at("hfhub.model_id")) and config.at("metaconfig.preprocessing.unique_hfhub_model_id"):
        if "-peft" not in out_model_id and config.at("trainer.lora"):
            out_model_id += "-peft"
        if wandb_run_id := config.at("wandb.id"):
            out_model_id += f"-{wandb_run_id}"
        else:
            out_model_id += f"-{generate_time_id()}"
        config.set("hfhub.model_id", out_model_id)

    return config


# %% ../../../nbs/hf.transformers.experiment.ipynb 6
def make_datacollator(tokenizer, response_template: str | None, response_template_context: str | None = None):
    if not response_template:
        return None

    if response_template_context is None:
        log.info(f"Creating completion-only data collator with response template '{response_template}'")
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    else:
        log.info(f"Creating completion-only data collator with response template '{response_template}' and context '{response_template_context}'")
        response_template_with_context = response_template_context + response_template
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[len(response_template_context):] 
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    return data_collator

# %% ../../../nbs/hf.transformers.experiment.ipynb 7
def prepare_model_for_training(tokenizer, model):
    model_id = model.name_or_path.lower()

    if "llama-2" in model_id:
        from bellem.hf.transformers.llama2 import prepare_llama2_for_training

        log.info("Base model is a llama-2 model, preparing it for training.")
        prepare_llama2_for_training(tokenizer, model)
    
    elif "llama-3" in model_id:
        from bellem.hf.transformers.llama3 import prepare_llama3_for_training

        log.info("Base model is a llama-3 model, preparing it for training.")
        prepare_llama3_for_training(tokenizer, model)
    
    else:
        log.warning(f"Base model '{model_id}' is not a llama-2 or llama-3 model, no special preparation is done.")

# %% ../../../nbs/hf.transformers.experiment.ipynb 8
def calculate_token_counts(
    tokenizer,
    dataset: Dataset,
    *,
    messages_field: str = "messages",
):
    if messages_field not in dataset.column_names:
        raise ValueError(
            f"Dataset must have `{messages_field}` columns if `text_field` is not specified."
        )

    text_field = "text"
    dataset = dataset.map(
        lambda example: {
            text_field: tokenizer.apply_chat_template(
                example[messages_field],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
    )

    # Inspect token counts
    tokenized_train_ds = dataset.map(lambda examples: tokenizer(examples[text_field]), batched=True)
    token_counts = [len(input_ids) for input_ids in tokenized_train_ds["input_ids"]]
    log.info(f"Input token counts: min={min(token_counts)}, max={max(token_counts)}")
    return token_counts

# %% ../../../nbs/hf.transformers.experiment.ipynb 9
def fine_tune(config: NestedDict):
    from peft import LoraConfig

    # Base model
    pretrained_model_config = config["pretrained_model"]
    model_id = pretrained_model_config.pop("model_name_or_path")
    tokenizer, base_model = load_tokenizer_model(model_id, **pretrained_model_config)
    log.info(f"Loaded base model {model_id}")

    # Prepare model for training
    prepare_model_for_training(tokenizer, base_model)

    # Train dataset
    train_ds = load_datasets(config.at("dataset.train")).shuffle(seed=config.at("seed"))
    log.info(f"Loaded training dataset with {len(train_ds)} samples.")

    # Inspect token counts
    dataset_text_field = config.at("trainer.dataset_text_field")
    token_counts = calculate_token_counts(tokenizer, train_ds)
    log.info(f"Input token counts: min={min(token_counts)}, max={max(token_counts)}")

    # Supervised fine-tuning
    if config.at("trainer.max_seq_length") is None:
        config.set("trainer.max_seq_length", ceil(max(token_counts) / 8) * 8)
    max_seq_length = config.at("trainer.max_seq_length")
    log.info(f"Setting max_seq_length={max_seq_length}")

    peft_config = LoraConfig(**config.at("trainer.lora", {}))

    packing = config.at("trainer.packing", False)

    data_collator = make_datacollator(
        tokenizer,
        config.at("trainer.response_template"),
        config.at("trainer.response_template_context"),
    )

    training_args = TrainingArguments(
        output_dir="./results",
        **config.at("trainer.training_args"),
    )

    trainer = SFTTrainer(
        tokenizer=tokenizer,
        model=base_model,
        peft_config=peft_config,
        train_dataset=train_ds,
        dataset_text_field=dataset_text_field,
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

# %% ../../../nbs/hf.transformers.experiment.ipynb 10
def prepare_model_for_inference(tokenizer, model):
    model_id = model.name_or_path.lower()

    if "llama-2" in model_id:
        from bellem.hf.transformers.llama2 import prepare_llama2_for_inference

        log.info("Base model is a llama-2 model, preparing it for inference.")
        prepare_llama2_for_inference(tokenizer, model)
    
    elif "llama-3" in model_id:
        from bellem.hf.transformers.llama3 import prepare_llama3_for_inference

        log.info("Base model is a llama-3 model, preparing it for inference.")
        prepare_llama3_for_inference(tokenizer, model)
    
    else:
        log.warning(f"Base model '{model_id}' is not a llama-2 or llama-3 model, no special preparation is done.")

# %% ../../../nbs/hf.transformers.experiment.ipynb 11
def _load_tokenizer_model(config: NestedDict):
    model_id = config.at("hfhub.model_id")
    kwargs = deepcopy(config.get("pretrained_model", {}))
    kwargs.pop("model_name_or_path", None)
    return load_tokenizer_model(model_id, **kwargs)


def make_pipeline(config, tokenizer, model):
    prepare_model_for_inference(tokenizer, model)
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **config.at("inference.pipeline", {}),
    )

def predict(
    config,
    *,
    tokenizer=None,
    model=None,
):
    # Load validation dataset
    dataset_config = config.at("dataset.validation")
    assert dataset_config, "Validation dataset is not provided!"
    dataset = load_datasets(dataset_config)
    assert len(dataset) > 0, "Validation dataset is empty!"

    # Ensure the dataset has input/output columns
    cols = dataset[0].keys()
    if "input" not in cols:
        if "messages" not in dataset.column_names:
            raise ValueError("Dataset must have `messages` column if `input` column are not provided.")
        dataset = dataset.map(partition_input_output_messages).remove_columns("messages")

    # Prepare text generation pipeline
    if tokenizer is None or model is None:
        tokenizer, model = _load_tokenizer_model(config)

    # Set up pipeline
    generation_params = config.at("inference.generation_params", {})
    if "max_new_tokens" not in generation_params:
        if "output" in dataset.column_names:
            token_counts = calculate_token_counts(tokenizer, dataset, messages_field = "output")
            log.info(f"Output token counts: min={min(token_counts)}, max={max(token_counts)}")
            generation_params["max_new_tokens"] = ceil(max(token_counts) / 8) * 8
        else:
            log.info("max_new_tokens is not set.")

    pipe = make_pipeline(config, tokenizer, model)

    # Generate
    # Create chats so that transformers library doesn't override our ListDataset
    chats = ListDataset([Chat(messages) for messages in dataset['input']])
    generations = generate(
        pipe,
        chats,
        **generation_params,
    )

    # Create dataframe 
    dataf = dataset.to_pandas()
    dataf["generation"] = generations
    return dataf

# %% ../../../nbs/hf.transformers.experiment.ipynb 12
def evaluate_(
    config,
    *,
    tokenizer=None,
    model=None,
    metric_kwargs: dict | None = None,
    output_parse_fn: Callable[[str], Any] | None = None,
):
    import evaluate

    output_parse_fn = output_parse_fn or (lambda x: x)

    dataf = predict(config, tokenizer=tokenizer, model=model)

    # Parse texts
    dataf["prediction"] = dataf["generation"].map(output_parse_fn)
    dataf["reference"] = dataf["output"].map(lambda x: x[0]['content']).map(output_parse_fn)

    # Compute scores
    metric = evaluate.load(config.at("evaluation.metric"))
    metric_kwargs = metric_kwargs or {}
    scores = metric.compute(
        predictions=dataf["prediction"].values,
        references=dataf["reference"].values,
        **metric_kwargs,
    )

    return scores, dataf