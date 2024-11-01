import shutil
from pathlib import Path

from bellem.ml.data import IMAGENET_LABEL_IDS


class ClassCopier:
    def __init__(self, source_dir, dest_dir):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(exist_ok=True, parents=True)

    def cp(self, class_id):
        print(f"Starting copying {class_id}")
        shutil.copytree(self.source_dir / class_id, self.dest_dir / class_id)
        print(f"Completed copying {class_id}")


def main(src, dst, label_ids):
    copier = ClassCopier(src, dst)
    for label_id in label_ids:
        copier.cp(label_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument("--dst")
    parser.add_argument("-n", type=int, required=False, default=None)
    parser.add_argument("--labelfile", type=str, required=False, default=None)
    args = parser.parse_args()

    if args.labelfile:
        with open(args.labelfile) as f:
            label_ids = [line.strip() for line in f.readlines()]
    elif args.n:
        label_ids = IMAGENET_LABEL_IDS[: args.n]
    else:
        raise RuntimeError("Either -n or --labelfile must be specified")

    main(args.src, args.dst, label_ids)
