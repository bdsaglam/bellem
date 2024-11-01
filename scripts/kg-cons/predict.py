import wandb

from bellem.hf.transformers.experiment import predict, preprocess_config
from bellem.logging import get_logger
from bellem.ml.experiment import main
from bellem.utils import NestedDict

log = get_logger(__name__)


def run_experiment(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)
    if not wandb.run.resumed:
        config = preprocess_config(config)

    pred_df = predict(config)
    wandb_run.log(
        {
            "prediction-dataframe": wandb.Table(dataframe=pred_df.reset_index()),
        }
    )
    pred_df.to_json('jerx-inferences.jsonl', lines=True, orient='records')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    main(run_experiment, args)
