import json
from difflib import SequenceMatcher
from pathlib import Path

import evaluate
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
        answers = [row["answer"], *row["answer_aliases"]]
        return any(match(answer, row["predicted_answer"]) for answer in answers)

    return func


def calculate_musique_scores(dataf: pd.DataFrame) -> dict:
    metric = evaluate.load("bdsaglam/musique")
    predictions = dataf["predicted_answer"].tolist()
    references = dataf.apply(lambda row: [row["answer"], *row["answer_aliases"]], axis=1).tolist()
    scores = metric.compute(predictions=predictions, references=references)
    return scores


def main(
    dataset_file: Path = typer.Option(...),
    answers_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    df = pd.read_json(dataset_file, orient="records", lines=True)

    answer_df = pd.read_json(answers_file, orient="records", lines=True)
    answer_df["predicted_answer"] = answer_df["question_decomposition"].map(lambda x: x[-1]["answer"])

    comp_df = pd.merge(df, answer_df[["id", "predicted_answer"]], on="id", how="inner")

    scores = calculate_musique_scores(comp_df)
    scores["fuzzy_match"] = comp_df.apply(lambda row: is_correct(fuzzy_match)(row), axis=1).mean()

    out.mkdir(exist_ok=True, parents=True)
    comp_df.to_json(out / "comparisons.jsonl", orient="records", lines=True)
    with open(out / "scores.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    typer.run(main)
