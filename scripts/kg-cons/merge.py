import json

import torch
from peft import AutoPeftModelForCausalLM

from bellek.logging import get_logger
from bellek.utils import NestedDict

log = get_logger()


def main(args):
    with open(args.cfg) as f:
        config = NestedDict.from_flat_dict(json.load(f))

    model_id = config.at("hfhub.model_id")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()
    model.push_to_hub(f"{model_id}-merged")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(args)
