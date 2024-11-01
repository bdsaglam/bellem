from bellem.hf.transformers.experiment import fine_tune, preprocess_config
from bellem.logging import get_logger
from bellem.ml.experiment import main
from bellem.utils import NestedDict, flatten_dict

log = get_logger(__name__)


def run_experiment(wandb_run):
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)
    trainer = fine_tune(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    main(run_experiment, args)
