import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import tenacity
import typer
from dotenv import load_dotenv
from tqdm import tqdm

from bellek.qdecomp.llm import make_question_decomposer

load_dotenv()


@cache
def get_qdecomposer():
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_random_exponential(multiplier=1, max=30),
    )(make_question_decomposer(model="llama3-70b-togetherai"))


def process_line(line):
    example = json.loads(line)
    sub_questions = get_qdecomposer()(question=example["question"])
    return {
        "id": example["id"],
        "question": example["question"],
        "question_decomposition": [
            {**qd, "question": pred_q} for qd, pred_q in zip(example["question_decomposition"], sub_questions)
        ],
    }


def process_lines(lines):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_line, line) for line in lines]
        for future in tqdm(as_completed(futures), total=len(lines), desc="Decomposing questions"):
            yield future.result()


def main(dataset_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(dataset_file) as f:
        lines = f.readlines()

    with open(out, "w") as dst:
        for result in process_lines(lines):
            dst.write(json.dumps(result, ensure_ascii=False))
            dst.write("\n")


if __name__ == "__main__":
    typer.run(main)
