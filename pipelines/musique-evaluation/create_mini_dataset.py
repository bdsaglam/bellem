import json
from pathlib import Path

import typer

from bellek.musique.constants import ABLATION_RECORD_IDS


def main(dataset_filepath: Path = typer.Argument(), out: Path = typer.Option(...)):
    with open(dataset_filepath, "r") as src, open(out, "w") as dst:
        for line in src:
            record = json.loads(line)
            if record["id"] in ABLATION_RECORD_IDS:
                dst.write(line.strip() + "\n")


if __name__ == "__main__":
    typer.run(main)
