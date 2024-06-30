import json
from pathlib import Path

import typer
from datasets import load_dataset
from rich.console import Console

from bellek.mhqa.llm import make_mhqa_chat

err = Console(stderr=True).print

HF_HUB_USER_NAME = "bdsaglam"


def make_mhqa_chat_example(example):
    context = "\n\n".join(p["paragraph_text"] for p in example["paragraphs"])
    question = example["question"]
    messages = make_mhqa_chat(context, question)
    return {"messages": messages}


def main(config_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(config_file) as f:
        dataset_config = json.load(f)

    ds_name = dataset_config["path"].split("/", 1)[-1]

    dsd = load_dataset(**dataset_config)

    mhqa_chat_dsd = dsd.map(make_mhqa_chat_example)
    mhqa_chat_dsd.push_to_hub(f"{HF_HUB_USER_NAME}/{ds_name}-mhqa-chat")

    for split, ds in mhqa_chat_dsd.items():
        ds.to_json(out / f"mhqa-chat-dataset-{split}.jsonl")


if __name__ == "__main__":
    typer.run(main)
