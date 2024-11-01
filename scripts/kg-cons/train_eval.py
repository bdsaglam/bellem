import pandas as pd
import wandb

from bellem.hf.transformers.experiment import evaluate_, fine_tune, preprocess_config
from bellem.jerx.eval import parse_triplet_strings
from bellem.logging import get_logger
from bellem.ml.experiment import main
from bellem.text.utils import fuzzy_match
from bellem.utils import NestedDict, flatten_dict

log = get_logger(__name__)


def log_evaluation_data(wandb_run, df: pd.DataFrame):
    name = "evaluation-dataframe"
    artifact = wandb.Artifact(name=name, type="dataset")
    filename = name + ".parquet"
    df.to_parquet(filename)
    artifact.add_file(local_path=filename)
    wandb_run.log_artifact(artifact)


def run_experiment(wandb_run):
    # prepare config
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)

    # fine-tuning
    trainer = fine_tune(config)

    # evaluation
    config = NestedDict.from_flat_dict(wandb_run.config)
    scores, eval_df = evaluate_(
        config,
        tokenizer=trainer.tokenizer,
        model=trainer.model,
        metric_kwargs={"eq_fn": lambda a, b: fuzzy_match(a, b, threshold=0.8)},
        output_parse_fn=parse_triplet_strings,
    )
    wandb_run.log(scores)
    log_evaluation_data(wandb_run, eval_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    main(run_experiment, args)
