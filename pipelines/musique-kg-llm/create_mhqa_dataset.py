import json
from pathlib import Path

import typer
from datasets import load_dataset
from rich.console import Console

err = Console(stderr=True).print


def is_question_about_record_label(example):
    question = example["question"]
    keywords = ["album", "record label"]
    return any(keyword in question.lower() for keyword in keywords)


def main(config_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(config_file) as f:
        dataset_config = json.load(f)
    ds = load_dataset(**dataset_config)
    ds = ds.filter(lambda example: not is_question_about_record_label(example))
    ds.to_json(out)

    name = "bdsaglam/musique-answerable-2hop-subset"
    ds.push_to_hub(name, split="train")


if __name__ == "__main__":
    typer.run(main)
