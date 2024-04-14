import typer

import wandb
from bellek.hf.transformers.experiment import evaluate_
from bellek.jerx.eval import parse_triplet_strings
from bellek.logging import get_logger
from bellek.text.utils import fuzzy_match
from bellek.utils import NestedDict

log = get_logger(__name__)


def run(config):
    return evaluate_(
        config,
        metric_kwargs={"eq_fn": lambda a, b: fuzzy_match(a, b, threshold=0.8)},
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
