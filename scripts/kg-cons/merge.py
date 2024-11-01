import json

from bellem.hf.transformers.utils import merge_adapters_and_publish
from bellem.utils import NestedDict


def main(args):
    with open(args.cfg) as f:
        config = NestedDict.from_flat_dict(json.load(f))
    model_id = config.at("hfhub.model_id")
    torch_dtype = config.at("pretrained_model.torch_dtype", "float16")
    merged_model_id = model_id.replace("-peft", "")
    return merge_adapters_and_publish(model_id, torch_dtype=torch_dtype, merged_model_id=merged_model_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(args)
