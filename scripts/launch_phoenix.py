from pathlib import Path

import phoenix as px
import typer
from dotenv import load_dotenv
from phoenix import TraceDataset
from phoenix.trace.utils import json_lines_to_df

load_dotenv()


PROMPT = """
Press
r: Refresh tracing data
q: Quit app
""".strip()


def get_trace_dataset(filepath: Path):
    with open(filepath) as f:
        lines = [line for line in f.readlines() if line.strip()]
    return TraceDataset(json_lines_to_df(lines))


def main(filepath: Path):
    px.launch_app(trace=get_trace_dataset(filepath))

    while True:
        output = input(PROMPT).lower()
        if output == "q":
            px.close_app()
            break
        if output == "r":
            px.close_app()
            px.launch_app(trace=get_trace_dataset(filepath))


if __name__ == "__main__":
    typer.run(main)
