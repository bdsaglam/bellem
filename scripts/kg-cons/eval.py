import typer
from datasets import load_dataset

import wandb
from bellek.logging import get_logger
from bellek.utils import NestedDict

log = get_logger(__name__)


def run(config):
    from bellek.jerx.eval import evaluate_model_jerx
    from bellek.hf.transformers.utils import load_tokenizer_model

    # Load validation dataset
    val_ds_config = config.at("dataset.validation")
    if val_ds_config is None:
        return
    val_ds = load_dataset(**val_ds_config)

    # Load model
    model_id = config.at("hfhub.model_id")
    quantization_config = config.at("pretrained_model.quantization_config")
    tokenizer, model = load_tokenizer_model(
        model_id,
        quantization_config=quantization_config,
    )
    if "llama" in model_id:
        from bellek.hf.transformers.llama import prepare_llama2_for_inference

        prepare_llama2_for_inference(tokenizer, model)

    response_template = config.at("trainer.response_template")

    return evaluate_model_jerx(
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
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(main)
