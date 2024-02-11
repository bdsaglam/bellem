import json
from difflib import SequenceMatcher
from operator import eq
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console

err = Console(stderr=True).print

load_dotenv()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def fuzzy_match(a, b, threshold=0.8):
    return (a in b) or (similarity(a, b) >= threshold)


def is_correct(match):
    def func(row):
        answers = [row["reference_answer"], *row["answer_aliases"]]
        return any(match(answer, row["predicted_answer"]) for answer in answers)
    return func


def main(
    dataset_file: Path = typer.Option(...),
    answers_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    df = pd.read_json(dataset_file, orient="records", lines=True)
    answer_df = pd.read_json(answers_file, orient="records", lines=True)
    comp_df = pd.DataFrame(
        {
            "id": df["id"],
            "question": df["question"].values,
            "reference_answer": df["answer"].values,
            "answer_aliases": df["answer_aliases"].values,
            "predicted_answer": answer_df["question_decomposition"].map(lambda x: x[-1]["answer"]).values,
        }
    )
    comp_df["exact_match"] = comp_df.apply(lambda row: is_correct(eq)(row), axis=1)
    comp_df["fuzzy_match"] = comp_df.apply(lambda row: is_correct(fuzzy_match)(row), axis=1)
    scores = dict(comp_df[["exact_match", "fuzzy_match"]].mean())

    out.mkdir(exist_ok=True, parents=True)
    comp_df.to_json(out / "comparisons.jsonl", orient="records", lines=True)
    with open(out / "scores.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    typer.run(main)
