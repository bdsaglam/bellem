from pathlib import Path

import pandas as pd
import typer
from datasets import Dataset
from dotenv import load_dotenv
from rich.console import Console
from transformers import AutoTokenizer

err = Console(stderr=True).print

load_dotenv()


system_prompt = """You are helpful assistant that grades the output for joint entity relation extraction task. The entity-relation-entity triplets are provided under `Triplets` section. Consider whether the triplets include all useful entity-relations in the provided `Text`. The grade is either 0 or 1. Your response must be in the following format:
GRADE: 0"""


def format_jerx_task(example):
    inp = example["jerx.input"]
    output = example["jerx.output"]
    return f"# Text\n{inp}\n\n# Triplets\n{output}"


def format_reward(example):
    return f"GRADE: {example['reward']}"


def make_reward_modelling_chat(example):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_jerx_task(example)},
            {"role": "assistant", "content": format_reward(example)},
        ]
    }


# LLAMA2 prompt format
llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-chat-hf", trust_remote_code=True)
llama2_tokenizer.pad_token = llama2_tokenizer.eos_token
llama2_tokenizer.padding_side = "right"


def format_for_llama2(example):
    text = llama2_tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}


def publish_datasets(reward_df: pd.DataFrame):
    prefix = "bdsaglam/musique-answerable-2hop-subset-jerx-reward"

    # reward dataset
    reward_ds = Dataset.from_pandas(reward_df)
    reward_ds_name = f"{prefix}"
    reward_ds.push_to_hub(reward_ds_name)

    # reward dataset - openai chat format
    openai_ds = reward_ds.map(make_reward_modelling_chat, remove_columns=["reward"])
    openai_ds_name = f"{prefix}-openai"
    openai_ds.push_to_hub(openai_ds_name)

    # reward dataset - llama2 format
    llama2_ds = openai_ds.map(format_for_llama2, remove_columns=["messages"])
    llama2_ds_name = f"{prefix}-llama2"
    llama2_ds.push_to_hub(llama2_ds_name)

    return [reward_ds_name, openai_ds_name, llama2_ds_name]


def main(
    reward_dataset_path: str = typer.Option(...),
    out: Path = typer.Option(...),
):
    reward_df = pd.read_json(reward_dataset_path, orient="records", lines=True)
    dataset_names = publish_datasets(reward_df)
    out.write_text("\n".join(dataset_names))


if __name__ == "__main__":
    typer.run(main)
