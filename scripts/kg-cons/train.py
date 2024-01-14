from bellek.hf.transformers.experiment import preprocess_config, fine_tune
from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.utils import NestedDict, flatten_dict, generate_time_id

log = get_logger(__name__)


def run_experiment(wandb_run):
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))

    # Generate unique model id
    model_id = config.at("hfhub.model_id")
    if config.at("trainer.lora"):
        model_id += "-peft"
    if "debug" not in model_id:
        model_id += f"-{generate_time_id()}"
    config.set("hfhub.model_id", model_id)

    wandb_run.config.update(flatten_dict(config), allow_val_change=True)

    trainer = fine_tune(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    main(run_experiment, args)
