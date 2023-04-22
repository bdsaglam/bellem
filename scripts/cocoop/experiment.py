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
from fastai.learner import Recorder
from fastai.data.all import *
from fastai.distributed import *
from fastai.torch_core import set_seed
from fastai.vision.all import *
import contextlib

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


def simple_label_func(fname):
    return imagenet_label_map[Path(fname).parent.name]


imagenet_synset_map = get_imagenet_id_synset_map()


def synset_label_func(fname):
    words = imagenet_synset_map[Path(fname).parent.name].split(",")
    return ",".join(words[:2])


def get_image_files_subset(path, folders=None):
    result = []
    for p in get_image_files(path):
        if folders and p.parent.name in folders:
            result.append(p)
    return result


def make_imagenet_sketch_dls(config):
    path = config.at("data.imagenet_sketch.path")
    valid_pct = config.at("data.imagenet_sketch.valid_pct", 0.3)
    batch_size = config.at("data.imagenet_sketch.batch_size", 64)
    clip_model_name = config.at("clip.model_name")
    device = config.at("device")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(load_clip_preprocess(clip_model_name))
    label_func = (
        simple_label_func if config.at("data.imagenet_sketch.labelling.kind") == "simple" else synset_label_func
    )
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=RandomSplitter(valid_pct=valid_pct),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    dls = dblock.dataloaders(path, bs=batch_size)
    return dls


def make_imagenet_dls(config):
    clip_model_name = config.at("clip.model_name")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(load_clip_preprocess(clip_model_name))
    label_func = simple_label_func if config.at("data.imagenet.labelling.kind") == "simple" else synset_label_func
    # subset = sorted(list(imagenet_label_map.keys()))[:100]
    # get_items = lambda path: get_image_files_subset(path, folders = subset)
    get_items = get_image_files
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_items,
        get_y=label_func,
        splitter=GrandparentSplitter(train_name="train", valid_name="val"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    path = config.at("data.imagenet.path")
    batch_size = config.at("data.imagenet.batch_size", 64)
    device = config["device"]
    dls = dblock.dataloaders(path, bs=batch_size)
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
    cbs = [WandbCallback(log_preds=False)]
    if config.at("train.early_stop"):
        cbs.append(EarlyStoppingCallback(patience=config.at("train.early_stop.patience")))
    model = prepare_prompt_learning_clip(
        make_prompt_learning_clip(
            class_names,
            clip_model_name=config.at("clip.model_name"),
            prec=config.at("clip.prec"),
            **config["cocoop"],
        )
    )
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        cbs=cbs,
    )
    if config.at("train.precision") == "fp16":
        learn = learn.to_fp16()
    n_epoch = config.at("train.n_epoch")
    lr = config.at("train.lr")
    training_ctx_mgr = learn.distrib_ctx() if config.at("train.distributed") else contextlib.nullcontext()
    with training_ctx_mgr:
        with learn.no_bar():
            learn.fit(n_epoch, lr)

    # evaluation
    print("Evaluating model on validation set")
    clf_summary = evaluate_slmc(learn, dls=dls, show=False)
    accuracy_score = clf_summary.loc["accuracy"][0]
    wandb_run.log(
        {
            "classification-summary": wandb.Table(dataframe=clf_summary.reset_index()),
            "accuracy": accuracy_score,
        }
    )


def main(args):
    with open(args.cfg) as f:
        config = prepare_config(NestedDict(json.load(f)))
    wandb_params = config.pop("wandb")
    with wandb.init(config=flatten_dict(config), **wandb_params) as wandb_run:
        run_experiment(wandb_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()
    main(args)
