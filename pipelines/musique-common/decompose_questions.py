import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import typer
from dotenv import load_dotenv
from tqdm import tqdm

from bellek.qdecomp.llm import make_question_decomposer

load_dotenv()


@cache
def get_qdecomposer():
    return make_question_decomposer(model="llama-3-70b-tgi")


def process_line(line):
    example = json.loads(line)
    sub_questions = get_qdecomposer()(question=example["question"])
    return {
        "id": example["id"],
        "question": example["question"],
        "question_decomposition": [{"question": q} for q in sub_questions],
    }


def process_lines(lines):
    with ProcessPoolExecutor(max_workers=8) as executor:
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
