import json
from pathlib import Path

import typer
from datasets import load_dataset
from rich.console import Console

err = Console(stderr=True).print

HF_HUB_USER_NAME = "bdsaglam"


def main(config_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(config_file) as f:
        dataset_config = json.load(f)

    ds_name = dataset_config["path"].split("/", 1)[-1]

    dsd = load_dataset(**dataset_config)
    for split, ds in dsd.items():
        ds = ds.filter(lambda example: example['answerable'])
        ds.to_json(out / f"base-dataset-{split}.jsonl")


if __name__ == "__main__":
    typer.run(main)
