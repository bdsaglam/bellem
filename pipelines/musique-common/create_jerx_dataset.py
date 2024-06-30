import json
from pathlib import Path

import pandas as pd
import typer
from datasets import Dataset, DatasetDict, load_dataset
from rich.console import Console

from bellek.jerx.fewshot.llm import DEFAULT_FEW_SHOT_EXAMPLE_MESSAGES, DEFAULT_JERX_SYSTEM_MESSAGE_FOR_LLAMA

err = Console(stderr=True).print

HF_HUB_USER_NAME = "bdsaglam"


def flatten_paragraphs(example):
    for paragraph in example["paragraphs"]:
        yield {
            "id": example["id"],
            "paragraph_idx": paragraph["idx"],
            "paragraph_text": paragraph["paragraph_text"],
            "paragraph_title": paragraph["title"],
            "is_supporting": paragraph["is_supporting"],
        }


def make_doc(row):
    return f"# {row['paragraph_title']}\n{row['paragraph_text']}"


def make_messages(text):
    return [
        dict(role="system", content=DEFAULT_JERX_SYSTEM_MESSAGE_FOR_LLAMA),
        *DEFAULT_FEW_SHOT_EXAMPLE_MESSAGES,
        dict(role="user", content=text),
    ]


def make_jerx_chat_dataset(dataset):
    df = pd.DataFrame([record for example in dataset for record in flatten_paragraphs(example)])
    df["text"] = df.apply(make_doc, axis=1)
    df["messages"] = df["text"].apply(make_messages)
    return Dataset.from_pandas(df)


def main(config_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(config_file) as f:
        dataset_config = json.load(f)

    dsd = load_dataset(**dataset_config)

    ds_name = dataset_config["path"].split("/", 1)[-1]
    jerx_ds_name = f"{ds_name}-paragraph-jerx-chat"

    jerx_chat_dsd = DatasetDict({split: make_jerx_chat_dataset(ds) for split, ds in dsd.items()})
    for split, ds in jerx_chat_dsd.items():
        ds.to_json(out / f"jerx-dataset-{split}.jsonl")

    jerx_chat_dsd.push_to_hub(f"{HF_HUB_USER_NAME}/{jerx_ds_name}")

    jerx_chat_dsd.filter(lambda example: example["is_supporting"]).push_to_hub(
        f"{HF_HUB_USER_NAME}/{jerx_ds_name}-supporting"
    )


if __name__ == "__main__":
    typer.run(main)
