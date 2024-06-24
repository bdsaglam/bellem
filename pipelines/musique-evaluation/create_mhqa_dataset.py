import json
from pathlib import Path

import pandas as pd
import typer
from datasets import Dataset, load_dataset
from rich.console import Console

from bellek.jerx.fewshot.llm import DEFAULT_JERX_CHAT_TEMPLATE_FOR_LLAMA

err = Console(stderr=True).print


def is_about_record_label(example):
    document = " ".join([p["paragraph_text"] for p in example["paragraphs"] if p["is_supporting"]])
    question = example["question"]
    keywords = ["record label"]
    return any(keyword in question.lower() or keyword in document for keyword in keywords)


def flatten_paragraphs(example):
    return [
        {
            "id": example["id"],
            "paragraph_idx": paragraph["idx"],
            "paragraph_text": paragraph["paragraph_text"],
            "paragraph_title": paragraph["title"],
            "is_supporting": paragraph["is_supporting"],
        }
        for paragraph in example["paragraphs"]
    ]


def make_doc(row):
    return f"# {row['paragraph_title']}\n{row['paragraph_text']}"


def make_messages(text):
    return [
        {"role": msg.role.value, "content": msg.content}
        for msg in DEFAULT_JERX_CHAT_TEMPLATE_FOR_LLAMA.format_messages(text=text)
    ]


def make_jerx_chat_dataset(dataset):
    df = pd.DataFrame(
        [record for example in dataset for record in flatten_paragraphs(example) if record["is_supporting"]]
    )
    df["text"] = df.apply(make_doc, axis=1)
    df.drop(columns=["paragraph_title", "paragraph_text"], inplace=True)
    df.drop_duplicates(inplace=True, subset=["text"])
    df["messages"] = df["text"].apply(make_messages)
    return Dataset.from_pandas(df)


def main(config_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(config_file) as f:
        dataset_config = json.load(f)
    ds = load_dataset(**dataset_config)
    ds.filter(lambda example: example["id"] not in {"2hop__389778_78303", "2hop__613779_55984", "2hop__590631_110882"})
    ds.to_json(out)

    user_name = "bdsaglam"
    ds_name = "musique-answerable-2hop"
    ds.push_to_hub(f"{user_name}/{ds_name}", split=dataset_config["split"])

    jerx_chat_ds = make_jerx_chat_dataset(ds)
    suffix = "-paragraph-jerx-chat"
    jerx_chat_ds_name = ds_name + suffix
    jerx_chat_ds.to_json(out.with_stem(out.stem + suffix))
    jerx_chat_ds.push_to_hub(f"{user_name}/{jerx_chat_ds_name}", split=dataset_config["split"])


if __name__ == "__main__":
    typer.run(main)
