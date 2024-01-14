import typer

import wandb
from bellek.hf.transformers.experiment import evalu8
from bellek.jerx.eval import parse_triplet_strings
from bellek.logging import get_logger
from bellek.utils import NestedDict

log = get_logger(__name__)


def run(config):
    return evalu8(
        config,
        output_parse_fn=parse_triplet_strings,
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
