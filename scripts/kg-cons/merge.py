import json

from bellek.ml.transformers import merge_adapters_and_publish
from bellek.utils import NestedDict


def main(args):
    with open(args.cfg) as f:
        config = NestedDict.from_flat_dict(json.load(f))
    model_id = config.at("hfhub.model_id")
    return merge_adapters_and_publish(model_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(args)
