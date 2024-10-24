import json
import os
import pandas as pd
import typer
from pathlib import Path

import dspy
from dspy.evaluate import Evaluate
from datasets import load_dataset
from bellek.utils import set_seed
from bellek.musique.eval import (
    aggregate_scores,
    calculate_metrics,
    compare_answers,
    compute_scores,
    compute_scores_dataframe,
)
from dotenv import load_dotenv
from rich.console import Console

print = Console(stderr=True).print

load_dotenv()

set_seed(89)

app = typer.Typer()


def configure_lm(model, temperature):
    lm = dspy.LM(
        "openai/" + model,
        temperature=temperature,
        cache=False,
        api_base=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    dspy.configure(lm=lm)


def format_paragraph(paragraph):
    text = paragraph["paragraph_text"]
    title = paragraph["title"]
    return f"# {title}\n{text}"


def make_example(record):
    supporting_paragraphs = [p for p in record["paragraphs"] if p["is_supporting"]]
    context = "\n\n".join([format_paragraph(p) for p in supporting_paragraphs])
    return dspy.Example(
        id=record["id"],
        question=record["question"],
        context=context,
        answer=record["answer"],
        answers=[record["answer"], *record["answer_aliases"]],
    ).with_inputs("question", "context")


class GenerateAnswer(dspy.Signature):
    """Answer the question based on the given context."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class QAModule(dspy.Module):
    def __init__(self, predict_cls=dspy.Predict):
        super().__init__()
        self.generate_answer = predict_cls(GenerateAnswer)

    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)


def get_predict_cls(technique):
    if technique == "standard":
        return dspy.Predict
    elif technique == "cot":
        return dspy.ChainOfThought
    elif technique == "cte":
        from bellek.dspy.predict.cte import ConnectTheEntities

        return ConnectTheEntities
    else:
        raise ValueError(f"Unknown technique: {technique}")


def evaluate_answer(example, pred, trace=None):
    scores = compute_scores(pred.answer, example.answers)
    return scores["f1"]


def dynamic_import(module, name):
    import importlib

    return getattr(importlib.import_module(module), name)


def make_optimizer(optimizer_path: Path):
    with open(optimizer_path) as f:
        optimizer_config = json.load(f)
    cls = dynamic_import("dspy.teleprompt", optimizer_config["class"])
    kwargs = optimizer_config["params"]
    if optimizer_config["with_metric"]:
        kwargs["metric"] = evaluate_answer
    return cls(**kwargs)


def preprocess_result(result):
    example, pred, score = result
    predictions = {f"predicted_{k}": v for k, v in dict(pred).items()}
    return {**dict(example), **predictions, "score": float(score)}


def make_results_dataframe(results):
    dataf = pd.json_normalize([preprocess_result(result) for result in results])
    return compute_scores_dataframe(dataf)


@app.command("evaluate")
def evaluate_main(
    dataset_path: str = typer.Option(..., help="Path to the dataset"),
    dataset_name: str = typer.Option(..., help="Name of the dataset"),
    dataset_split: str = typer.Option(..., help="Dataset split to use (e.g., 'train', 'validation')"),
    model: str = typer.Option(..., help="Name of the model to use"),
    temperature: float = typer.Option(..., help="Temperature parameter for the model"),
    technique: str = typer.Option(..., help="Prompting technique to use"),
    load_from: str = typer.Option(default="UNSET", help="Path to a saved model to load"),
    out: Path = typer.Option(..., help="Output directory for generated results"),
):
    out.mkdir(parents=True, exist_ok=True)

    # Set up LLM
    configure_lm(model, temperature)

    # Load and preprocess datasets
    ds = load_dataset(dataset_path, dataset_name, split=dataset_split)
    examples = [make_example(record) for record in ds]
    print(f"Loaded {len(examples)} examples")

    # Create the program
    program = QAModule(predict_cls=get_predict_cls(technique))
    is_trained = False
    if load_from and load_from != "UNSET":
        print(f"Loading model from {load_from}")
        program.load(load_from)
        is_trained = True

    # Evaluate the program
    evaluate_program = Evaluate(
        metric=evaluate_answer,
        devset=examples,
        num_threads=16,
        display_progress=True,
        return_outputs=True,
    )
    _, results = evaluate_program(program)
    result_df = make_results_dataframe(results)
    scores = aggregate_scores(result_df)

    # Save the results
    result_df.to_json(out / "results.json", orient="records", lines=True)

    # Save the scores
    prefix = "post" if is_trained else "pre"
    with open(out / "scores.json", "w") as f:
        json.dump({prefix: scores}, f, indent=2)


@app.command("train")
def train_main(
    dataset_path: str = typer.Option(..., help="Path to the dataset"),
    dataset_name: str = typer.Option(..., help="Name of the dataset"),
    dataset_split: str = typer.Option(..., help="Dataset split to use (e.g., 'train', 'validation')"),
    model: str = typer.Option(..., help="Name of the model to use"),
    temperature: float = typer.Option(..., help="Temperature parameter for the model"),
    technique: str = typer.Option(..., help="Prompting technique to use"),
    load_from: str = typer.Option(default="UNSET", help="Path to a saved model to load"),
    optimizer_path: Path = typer.Option(..., help="Path to the optimizer config"),
    out: Path = typer.Option(..., help="Output file for trained program"),
):
    out.parent.mkdir(parents=True, exist_ok=True)

    # Set up LLM
    configure_lm(model, temperature)

    # Load and preprocess datasets
    ds = load_dataset(dataset_path, dataset_name, split=dataset_split)
    examples = [make_example(record) for record in ds]
    print(f"Loaded {len(examples)} examples")

    # Create the program
    program = QAModule(predict_cls=get_predict_cls(technique))
    if load_from and load_from != "UNSET":
        print(f"Loading model from {load_from}")
        program.load(load_from)

    # Train the program
    optimizer = make_optimizer(optimizer_path)
    trained_program = optimizer.compile(program, trainset=examples)

    # Save the trained program
    trained_program.save(out)


if __name__ == "__main__":
    app()
