import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

from openai import OpenAI
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

from bellem.musique.qa import make_qa_func
from bellem.musique.singlehop import BaselineSingleHop
from bellem.utils import set_seed

print = Console(stderr=True).print

load_dotenv()

# For reproducibility
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
    dataset_path: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    dataset_split: str = typer.Option(...),
    out: Path = typer.Option(...),
    model: str = typer.Option("gpt-3.5-turbo"),
    temperature: float = typer.Option(0.1),
    system_prompt_filepath: str = typer.Option(default="None"),
    user_prompt_template_filepath: str = typer.Option(default="None"),
    few_shot_examples_filepath: str = typer.Option(default="None"),
    ignore_errors: bool = typer.Option(False),
    resume: bool = typer.Option(False),
):
    out.mkdir(exist_ok=True, parents=True)

    examples = load_dataset(dataset_path, name=dataset_name, split=dataset_split)

    # Prepare the QA pipeline
    qa_kwargs = {}

    if system_prompt_filepath and system_prompt_filepath != "None":
        with open(system_prompt_filepath) as f:
            qa_kwargs["system_prompt"] = f.read().strip()

    if user_prompt_template_filepath and user_prompt_template_filepath != "None":
        with open(user_prompt_template_filepath) as f:
            qa_kwargs["user_prompt_template"] = f.read().strip()

    if few_shot_examples_filepath and few_shot_examples_filepath != "None":
        with open(few_shot_examples_filepath) as f:
            qa_kwargs["few_shot_examples"] = json.load(f)

    qa_func = make_qa_func(**qa_kwargs)
    openai_client = OpenAI(max_retries=3)
    qa_func = partial(qa_func, model_name=model, completion_kwargs={"temperature": temperature}, client=openai_client)

    qa_pipeline = BaselineSingleHop(qa_func, perfect_retrieval_func)

    # Process the samples
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
