import json
from pathlib import Path

import typer
from datasets import load_dataset


def main(config_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(config_file) as f:
        dataset_config = json.load(f)
    ds = load_dataset(**dataset_config)
    ds.to_json(out)


if __name__ == "__main__":
    typer.run(main)
