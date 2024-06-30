import json
from pathlib import Path

import pandas as pd
import typer
from datasets import Dataset
from dotenv import load_dotenv

from bellek.jerx.fewshot.llm import DEFAULT_FEW_SHOT_EXAMPLE_MESSAGES, DEFAULT_JERX_SYSTEM_MESSAGE_FOR_LLAMA
from bellek.utils import set_seed

load_dotenv()

set_seed(89)


HF_HUB_USER_NAME = "bdsaglam"


def make_jerx_text(example):
    return "\n\n".join(p["paragraph_text"] for p in example["paragraphs"] if p["is_supporting"])


def make_few_shot_chat(example):
    messages = [
        dict(role="system", content=DEFAULT_JERX_SYSTEM_MESSAGE_FOR_LLAMA),
        *DEFAULT_FEW_SHOT_EXAMPLE_MESSAGES,
        {"role": "assistant", "content": "\n".join(example["triplets"])},
    ]
    return {"messages": messages}


def main(
    dataset_file: Path = typer.Option(...),
    answer_comparison_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    DATA_DIR = dataset_file.parent

    ds_df = pd.read_json(dataset_file, orient="records", lines=True)
    comp_df = pd.read_json(answer_comparison_file, orient="records", lines=True)
    df = pd.merge(
        ds_df.drop(columns=["answerable", "answer", "answer_aliases"]),
        comp_df.drop(
            columns=[
                "answerable",
                "paragraphs",
                "question_decomposition",
                "question",
                "answer",
                "answer_aliases",
                "answers",
            ]
        ),
        on="id",
        suffixes=("", ""),
    )

    def load_triplets(example):
        id = example["id"]
        docs_filepath = DATA_DIR / f"knowledge-graphs/{id}/documents.jsonl"
        if not docs_filepath.exists():
            return []
        triplets = []
        with open(docs_filepath) as f:
            for line in f:
                doc = json.loads(line)
                triplets.extend(doc["triplets"])
        return [" | ".join(triplet) for triplet in triplets]

    df["triplets"] = df.apply(load_triplets, axis=1)
    df["text"] = df.apply(make_jerx_text, axis=1)

    mask = df["exact_match"]
    success_df = df.loc[mask]

    jerx_ds_name = "musique-answerable-2hop-jerx"

    jerx_ds = Dataset.from_list(
        [{"text": row["text"], "triplets": row["triplets"]} for _, row in success_df.iterrows()]
    )
    jerx_ds.push_to_hub(f"{HF_HUB_USER_NAME}/{jerx_ds_name}", split="train")

    jerx_chat_ds = jerx_ds.map(make_few_shot_chat, remove_columns=["text", "triplets"])
    jerx_chat_ds.push_to_hub(f"{HF_HUB_USER_NAME}/{jerx_ds_name}-chat", split="train")
    jerx_chat_ds.to_json(out / "jerx-chat.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    typer.run(main)
