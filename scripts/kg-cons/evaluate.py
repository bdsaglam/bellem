import os

import typer
from datasets import load_dataset

import wandb
from bellek.logging import get_logger
from bellek.ml.kg.cons import parse_triplet_strings
from bellek.ml.llama import prepare_llama2_for_inference
from bellek.ml.transformers import load_tokenizer_model
from bellek.utils import NestedDict

log = get_logger(__name__)


def before_experiment(wandb_run):
    os.environ["WANDB_PROJECT"] = wandb_run.project
    if seed := wandb_run.config.get("seed"):
        from fastai.torch_core import set_seed

        set_seed(seed)


def load_ds(dataset_config):
    return load_dataset(
        dataset_config["path"],
        dataset_config.get("name"),
        split=dataset_config.get("split"),
    )


def evaluate_model(wandb_run, config, tokenizer, model):
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
        max_new_tokens=config.at("evaluation.max_new_tokens", 256),
        batch_size=config.at("evaluation.batch_size", 2),
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


def run(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)

    before_experiment(wandb_run)

    model_id = config.at("hfhub.model_id")

    # Determine model class
    if "-peft" in model_id:
        from peft import AutoPeftModelForCausalLM

        auto_model_cls = AutoPeftModelForCausalLM
    else:
        from transformers import AutoModelForCausalLM

        auto_model_cls = AutoModelForCausalLM

    # Load model
    quantization_config = config.at("pretrained_model.quantization_config")
    tokenizer, model = load_tokenizer_model(
        model_id,
        auto_model_cls=auto_model_cls,
        quantization_config=quantization_config,
    )

    # Evaluate model
    evaluate_model(wandb_run, config, tokenizer, model)


def main(wandb_project: str, wandb_run_id: str):
    wandb_run = wandb.init(project=wandb_project, resume=wandb_run_id)
    try:
        run(wandb_run)
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(main)
