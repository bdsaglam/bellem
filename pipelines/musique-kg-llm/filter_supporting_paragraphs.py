import json
from pathlib import Path
import typer


def main(dataset_file: Path = typer.Option(...), out: Path = typer.Option(...)):
    with open(dataset_file) as src:
        with open(out, "w") as dst:
            for line in src:
                example = json.loads(line)
                example["paragraphs"] = [p for p in example["paragraphs"] if p["is_supporting"]]
                dst.write(json.dumps(example, ensure_ascii=False))
                dst.write("\n")


if __name__ == "__main__":
    typer.run(main)
