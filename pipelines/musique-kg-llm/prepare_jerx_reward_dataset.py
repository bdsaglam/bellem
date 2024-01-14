import json
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console

from bellek.utils import set_seed

err = Console(stderr=True).print

load_dotenv()

set_seed(42)


def read_llm_traces(filepath: Path | str) -> pd.DataFrame:
    with open(filepath) as f:
        data = [json.loads(line) for line in f if line.strip()]
    df = pd.json_normalize([item for item in data if item["name"] == "llm"])
    df["id"] = filepath.parent.name
    return df


def make_llm_text(row):
    llm_input = row["attributes.llm.prompts"][0]
    llm_output = row["attributes.output.value"]
    return llm_input + llm_output


def load_llm_erx_generations(knowledge_graph_directory: Path, example_ids: list[str]) -> pd.DataFrame:
    df = pd.concat([read_llm_traces(knowledge_graph_directory / id / "traces.jsonl") for id in example_ids])
    df["text"] = df.apply(make_llm_text, axis=1)
    return df[["id", "text"]]


def main(
    knowledge_graph_directory: Path = typer.Option(...),
    answer_comparisons_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    comparison_df = pd.read_json(answer_comparisons_file, orient="records", lines=True)
    llm_erx_df = load_llm_erx_generations(knowledge_graph_directory, comparison_df["id"].tolist())

    reward_df = pd.merge(llm_erx_df, comparison_df[["id", "fuzzy_match"]], on="id")
    reward_df["reward"] = reward_df["fuzzy_match"].astype(int)
    reward_df.drop(columns=["fuzzy_match"], inplace=True)

    reward_df.sample(frac=1).to_json(out, orient="records", lines=True)


if __name__ == "__main__":
    typer.run(main)
