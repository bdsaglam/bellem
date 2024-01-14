from bellek.hf.transformers.experiment import preprocess_config, train
from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.utils import NestedDict, flatten_dict

log = get_logger(__name__)


def run_experiment(wandb_run):
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)
    trainer = train(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    main(run_experiment, args)
