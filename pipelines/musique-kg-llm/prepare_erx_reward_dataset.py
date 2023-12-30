import json
from datasets import Dataset
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console

from bellek.utils import generate_time_id

err = Console(stderr=True).print

load_dotenv()


def read_traces(filepath: Path | str) -> pd.DataFrame:
    with open(filepath) as f:
        data = [json.loads(line) for line in f if line.strip()]
    df = pd.json_normalize(data)
    df["id"] = filepath.parent.name
    return df


def make_llm_text(row):
    llm_input = row["attributes.llm.prompts"][0]
    llm_output = row["attributes.output.value"]
    return llm_input + llm_output


def load_llm_erx_generations(knowledge_graph_directory) -> pd.DataFrame:
    trace_df = pd.concat([read_traces(fp) for fp in knowledge_graph_directory.glob("**/traces.jsonl")])
    llm_trace_df = trace_df[trace_df["name"] == "llm"].copy()
    llm_trace_df["text"] = llm_trace_df.apply(make_llm_text, axis=1)
    return llm_trace_df[["id", "text"]]


def main(
    knowledge_graph_directory: Path = typer.Option(...),
    answer_comparisons_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    llm_erx_df = load_llm_erx_generations(knowledge_graph_directory)
    comparison_df = pd.read_json(answer_comparisons_file, orient="records", lines=True)

    reward_df = pd.merge(llm_erx_df, comparison_df[["id", "fuzzy_match"]], on="id")
    reward_df["reward"] = reward_df["fuzzy_match"].astype(int)
    reward_df.drop(columns=["fuzzy_match"], inplace=True)

    reward_df.to_json(out, orient="records", lines=True)

    # Push to HF Hub
    dataset_name = f'bdsaglam/musique-answerable-2hop-subset-erx-reward-{generate_time_id()}'
    Dataset.from_pandas(reward_df, split='train').push_to_hub(dataset_name)


if __name__ == "__main__":
    typer.run(main)
