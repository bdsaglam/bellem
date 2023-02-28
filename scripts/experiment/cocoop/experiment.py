import json
import os
import warnings
from pathlib import Path

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

from bellek.ml.clip import load_clip_preprocess, make_tfms_from_clip_preprocess
from bellek.ml.cocoop import *
from bellek.ml.data import *
from bellek.ml.evaluation import evaluate_slmc
from bellek.ml.experiment import *
from bellek.ml.vision import *
from bellek.utils import *

imagenet_label_map = get_imagenet_id_label_map()

def label_func(fname):
    return imagenet_label_map[Path(fname).parent.name]

def make_imagenet_sketch_dls(config):
    path = config.at("data.imagenet_sketch.path")
    valid_pct = config.at("data.imagenet_sketch.valid_pct", 0.3)
    batch_size = config.at("data.imagenet_sketch.batch_size", 64)
    clip_model_name = config.at("clip.model_name", "RN50")
    device = config.at("device")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(
        load_clip_preprocess(clip_model_name)
    )
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=RandomSplitter(valid_pct=valid_pct),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    dls = dblock.dataloaders(path, bs=batch_size, device=device)
    return dls


def run_experiment(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)

    # dataloaders
    print("Creating dataloaders")
    dls = make_imagenet_sketch_dls(config)
    class_names = list(dls.vocab)
    print(f"#class: {len(class_names)}")

    # training
    print(f"Training on {config.get('device')}")
    cbs = [SaveModelCallback(), WandbCallback()]
    if config.at("train.early_stop"):
        cbs.append(
            EarlyStoppingCallback(patience=config.at("train.early_stop.patience"))
        )
    model = prepare_prompt_learning_clip(
        make_prompt_learning_clip(class_names, **config["cocoop"])
    )
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        cbs=cbs,
    )
    learn.fit(config.at("train.n_epoch"), config.at("train.lr"))

    # evaluation
    print("Evaluation model on validation set")
    clf_summary = evaluate_slmc(learn, dls=dls, show=False)
    wandb_run.log(
        {
            "classification-summary": wandb.Table(dataframe=clf_summary.reset_index()),
        }
    )


def main(args):
    with open(args.cfg) as f:
        config = prepare_config(NestedDict(json.load(f)))
    wandb_params = config.pop("wandb")
    with context_chdir(make_experiment_dir()):
        with wandb.init(config=flatten_dict(config), **wandb_params) as wandb_run:
            run_experiment(wandb_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(args)
