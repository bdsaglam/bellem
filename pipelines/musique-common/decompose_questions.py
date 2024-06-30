import json
from concurrent.futures import ProcessPoolExecutor
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
    total_lines = len(lines)  # Get the total number of lines
    progress_bar = tqdm(total=total_lines, desc="Processing lines")

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for line in lines:
            future = executor.submit(process_line, line)
            future.add_done_callback(lambda p: progress_bar.update())
            results.append(future)

    progress_bar.close()
    return results


def main(dataset_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(dataset_file) as f:
        lines = f.readlines()  # Read all lines from the source file

    results = process_lines(lines)

    with open(out, "w") as dst:
        for future in results:
            dst.write(json.dumps(future.result()), ensure_ascii=False)
            dst.write("\n")


if __name__ == "__main__":
    typer.run(main)
