import json
from datetime import datetime
from pathlib import Path

from bellek.musique.eval import compute_scores
import typer
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from bellek.utils import set_seed

print = Console(stderr=True).print

load_dotenv()

set_seed(89)


def process_example(
    example: dict,
    qa_result: dict,
    out: Path,
    resume: bool,
):
    example_id = example["id"]

    # Check if the example is already processed
    result_file = out / f"{example_id}.json"
    if resume and result_file.exists():
        print(f"Skipping the sample {example_id} because it is already processed.")
        return

    # Calculate the scores
    result_file.unlink(missing_ok=True)

    predicted_answer = qa_result.get("answer")
    reference_answers = example["answers"]
    scores = compute_scores(predicted_answer, reference_answers)
    result = {
        "predicted_answer": predicted_answer,
        "reference_answers": reference_answers,
        **scores,
    }

    with open(result_file, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


def main(
    dataset_file: Path = typer.Option(...),
    qa_dir: Path = typer.Option(...),
    out: Path = typer.Option(...),
    resume: bool = typer.Option(False),
):
    out.mkdir(exist_ok=True, parents=True)

    with open(dataset_file) as f:
        examples = [json.loads(line) for line in f]

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
            resume=resume,
        )

    with open(out / "timestamp.txt", "w") as f:
        f.write(datetime.now().isoformat())


if __name__ == "__main__":
    typer.run(main)