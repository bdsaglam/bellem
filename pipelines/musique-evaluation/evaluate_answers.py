import json
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console

from bellek.musique.eval import calculate_metrics, compare_answers

err = Console(stderr=True).print

load_dotenv()


def load_answers(answers_dir: Path, example_ids: list[str]) -> pd.DataFrame:
    answers = []
    for example_id in example_ids:
        try:
            with open(answers_dir / example_id / "answer.json") as f:
                answers.append(json.load(f))
        except FileNotFoundError:
            err(f"Answer file not found for id {example_id}")
    return pd.DataFrame(answers)

def main(
    dataset_file: Path = typer.Option(...),
    answers_dir: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    df = pd.read_json(dataset_file, orient="records", lines=True)

    answer_df = load_answers(answers_dir, df["id"].tolist())
    answer_df["predicted_answer"] = answer_df["question_decomposition"].map(lambda x: x[-1]["answer"])

    comp_df = pd.merge(df, answer_df[["id", "predicted_answer"]], on="id", how="inner")
    comp_df = compare_answers(comp_df)

    scores = calculate_metrics(comp_df)
    scores["fuzzy_match"] = comp_df["fuzzy_match"].mean()

    out.mkdir(exist_ok=True, parents=True)
    comp_df.to_json(out / "comparisons.jsonl", orient="records", lines=True)
    with open(out / "scores.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    typer.run(main)
