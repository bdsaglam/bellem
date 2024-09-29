from functools import partial
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

from bellek.musique.singlehop import BaselineSingleHop
import typer
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from bellek.utils import set_seed
from bellek.musique.qa import load_qa_func

print = Console(stderr=True).print

load_dotenv()

set_seed(89)


def perfect_retrieval_func(docs: list[dict], query: str):
    return [doc for doc in docs if doc["is_supporting"]]


def process_example(
    example: dict,
    qa_pipeline: Callable[[dict], dict],
    out: Path,
    ignore_errors: bool,
    resume: bool,
):
    example_id = example["id"]

    result_file = out / f"{example_id}.json"
    if resume and result_file.exists():
        print(f"Skipping the sample {example_id} because it is already processed.")
        return

    result_file.unlink(missing_ok=True)

    try:
        print(f"Answering the question in the sample {example_id}")
        result = qa_pipeline(example)
        with open(result_file, "w") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"Failed to answer the question for sample {example_id}.\n{exc}")
        if ignore_errors:
            return
        raise exc


def main(
    dataset_file: Path = typer.Option(...),
    out: Path = typer.Option(...),
    prompt: str = typer.Option("standard"),
    model: str = typer.Option("gpt-3.5-turbo"),
    temperature: float = typer.Option(0.1),
    ignore_errors: bool = typer.Option(False),
    resume: bool = typer.Option(False),
    subset: int = typer.Option(0),
):
    out.mkdir(exist_ok=True, parents=True)

    with open(dataset_file) as f:
        examples = [json.loads(line) for line in f]

    if subset:
        examples = examples[:subset]

    qa_func = load_qa_func(prompt)
    qa_func = partial(qa_func, model_name=model, completion_kwargs={"temperature": temperature})

    qa_pipeline = BaselineSingleHop(qa_func, perfect_retrieval_func)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_example,
                example=example,
                qa_pipeline=qa_pipeline,
                out=out,
                ignore_errors=ignore_errors,
                resume=resume,
            )
            for example in examples
        ]

        for future in tqdm(as_completed(futures), total=len(examples), desc="Answering questions"):
            future.result()

    with open(out / "timestamp.txt", "w") as f:
        f.write(datetime.now().isoformat())


if __name__ == "__main__":
    typer.run(main)
