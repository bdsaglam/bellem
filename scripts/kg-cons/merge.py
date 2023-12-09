import json

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from bellek.logging import get_logger
from bellek.utils import NestedDict

log = get_logger(__name__)


def main(args):
    with open(args.cfg) as f:
        config = NestedDict.from_flat_dict(json.load(f))

    model_id = config.at("hfhub.model_id")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = model.merge_and_unload()

    merged_model_id = f"{model_id}-merged"
    model.push_to_hub(merged_model_id)
    tokenizer.push_to_hub(merged_model_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(args)
