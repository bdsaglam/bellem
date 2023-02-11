import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from fastai.callback.wandb import *
from fastai.data.all import *
from fastai.torch_core import set_seed
from fastai.vision.all import *
from fastcore.basics import ifnone, store_attr
from torchvision import transforms

from bellek.ml.data import *
from bellek.ml.experiment import *
from bellek.ml.vision import *
from bellek.utils import *


def run_experiment(wandb_run):
    config = Tree.from_flat_dict(wandb_run.config)
    print("Config")
    print(config)

    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)

    print("Training...")
    wandb_run.log({"val_loss": 10})


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg")
    args = parser.parse_args()

    with open(args.cfg) as f:
        config = prepare_config(Tree(json.load(f)))

    with context_chdir(make_experiment_dir()):
        wandb_config = config["wandb"]
        with wandb.init(config=config.flat(), **wandb_config) as wandb_run:
            run_experiment(wandb_run)
