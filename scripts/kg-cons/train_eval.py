import wandb
from bellek.hf.transformers.experiment import evalu8, fine_tune, preprocess_config
from bellek.jerx.eval import parse_triplet_strings
from bellek.logging import get_logger
from bellek.ml.experiment import main
from bellek.text.utils import fuzzy_match
from bellek.utils import NestedDict, flatten_dict

log = get_logger(__name__)


def run_experiment(wandb_run):
    # prepare config
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)

    # fine-tuning
    trainer = fine_tune(config)

    # evaluation
    config = NestedDict.from_flat_dict(wandb_run.config)
    scores, eval_df = evalu8(
        config,
        tokenizer=trainer.tokenizer,
        model=trainer.model,
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
