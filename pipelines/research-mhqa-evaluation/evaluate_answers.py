import json
from datetime import datetime
from pathlib import Path

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from bellem.musique.eval import compute_scores
from bellem.utils import set_seed

print = Console(stderr=True).print

load_dotenv()

set_seed(89)


def process_example(
    example: dict,
    qa_result: dict,
    out: Path,
):
    example_id = example["id"]

    predicted_answer = qa_result.get("answer")
    reference_answers = list(set([example["answer"], *example["answer_aliases"]]))
    scores = compute_scores(predicted_answer, reference_answers)
    result = {
        "predicted_answer": predicted_answer,
        "reference_answers": reference_answers,
        **scores,
    }

    result_file = out / f"{example_id}.json"
    with open(result_file, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


def main(
    dataset_path: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    dataset_split: str = typer.Option(...),
    qa_dir: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    out.mkdir(exist_ok=True, parents=True)

    examples = load_dataset(dataset_path, name=dataset_name, split=dataset_split)

    examples_with_answers = []
    for example in examples:
        example_id = example["id"]
        qa_file = qa_dir / f"{example_id}.json"
        if not qa_file.exists():
            continue

        qa_result = json.loads(qa_file.read_text())
        examples_with_answers.append((example, qa_result))

    for example, qa_result in tqdm(examples_with_answers, desc="Evaluating answers"):
        process_example(
            example=example,
            qa_result=qa_result,
            out=out,
        )

    with open(out / "timestamp.txt", "w") as f:
        f.write(datetime.now().isoformat())


if __name__ == "__main__":
    typer.run(main)
