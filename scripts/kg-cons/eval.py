import wandb

from bellek.hf.transformers.experiment import evaluate_, preprocess_config
from bellek.jerx.eval import parse_triplet_strings
from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.text.utils import fuzzy_match
from bellek.utils import NestedDict

log = get_logger(__name__)


def run_experiment(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)
    if not wandb.run.resumed:
        config = preprocess_config(config)

    scores, eval_df = evaluate_(
        config,
        metric_kwargs={"eq_fn": lambda a, b: fuzzy_match(a, b, threshold=0.8)},
        output_parse_fn=parse_triplet_strings,
    )
    wandb_run.log(
        {
            **scores,
            "evaluation-dataframe": wandb.Table(dataframe=eval_df.reset_index()),
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    main(run_experiment, args)
