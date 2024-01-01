import re

import typer
from datasets import load_dataset

import wandb
from bellek.logging import get_logger
from bellek.utils import NestedDict

log = get_logger(__name__)


def load_ds(dataset_config):
    return load_dataset(
        dataset_config["path"],
        dataset_config.get("name"),
        split=dataset_config.get("split"),
    )


def parse_reward(text):
    line = text.split("\n")[0].strip()
    # match the number in the first line
    match = re.search(r"[-+]?\d*\.\d+|\d+", line)
    if match is None:
        raise ValueError(f"Could not parse reward from text: {text}")
    return float(match.group())


def evaluate_pipe_reward_pred(dataset, pipe):
    import evaluate

    tokenizer = pipe.tokenizer

    def _clean(text):
        return text.replace(tokenizer.special_tokens_map["eos_token"], "").strip()

    results = pipe(dataset["input"])
    generations = [result[0]["generated_text"] for result in results]
    predictions = [parse_reward(_clean(text)) for text in generations]
    references = [parse_reward(_clean(text)) for text in dataset["output"]]

    dataf = dataset.to_pandas()
    dataf["generation"] = generations
    dataf["prediction"] = predictions
    dataf["reference"] = references

    metric = evaluate.load("exact_match")
    scores = metric.compute(predictions=predictions, references=references)
    return scores, dataf


def evaluate_model_reward_pred(
    dataset,
    *,
    response_template: str,
    tokenizer,
    model,
    max_new_tokens=256,
    batch_size=4,
    **kwargs,
):
    assert len(dataset) > 0, "Dataset is empty!"

    def extract_input_output(example):
        input, output = example["text"].rsplit(response_template, 1)
        input += response_template
        return {"input": input, "output": output}

    dataset = dataset.map(extract_input_output)

    # setup generation pipeline
    from transformers import pipeline

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        return_full_text=False,
        **kwargs,
    )

    return evaluate_pipe_reward_pred(dataset, pipe)


def run(config):
    from bellek.ml.transformers import load_tokenizer_model
    from bellek.ml.transformers import preprocess_config as tpc

    # Setup config related to transformers library
    config = tpc(config)

    # Load validation dataset
    val_ds_config = config.at("dataset.validation")
    if val_ds_config is None:
        return
    val_ds = load_ds(val_ds_config)

    # Load model
    model_id = config.at("hfhub.model_id")
    quantization_config = config.at("pretrained_model.quantization_config")
    tokenizer, model = load_tokenizer_model(
        model_id,
        quantization_config=quantization_config,
    )
    if "llama" in model_id:
        from bellek.lang.llama import prepare_llama2_for_inference

        prepare_llama2_for_inference(tokenizer, model)

    response_template = config.at("trainer.response_template")

    return evaluate_model_reward_pred(
        val_ds,
        response_template=response_template,
        tokenizer=tokenizer,
        model=model,
        **config.at("evaluation", {}),
    )


def main(
    wandb_project: str = typer.Option(default=...),
    wandb_run_id: str = typer.Option(default=...),
):
    wandb_run = wandb.init(project=wandb_project, resume=wandb_run_id)
    try:
        config = NestedDict.from_flat_dict(wandb_run.config)
        scores, eval_df = run(config)
        wandb_run.log(
            {
                **scores,
                "evaluation-dataframe": wandb.Table(dataframe=eval_df.reset_index()),
            }
        )
        return scores, eval_df
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(main)
