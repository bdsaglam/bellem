import json
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
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from fastai.callback.wandb import *
from fastai.data.all import *
from fastai.torch_core import set_seed
from fastai.vision.all import *
from fastcore.basics import ifnone, store_attr
from torchvision import transforms

from bellek.ml.clip import *
from bellek.ml.cocoop import *
from bellek.ml.data import *
from bellek.ml.evaluation import evaluate_slmc
from bellek.ml.experiment import *
from bellek.ml.mcd import *
from bellek.ml.vision import *
from bellek.utils import *

imagenet_label_map = get_imagenet_id_label_map()


def simple_label_func(fname):
    return imagenet_label_map[Path(fname).parent.name]


imagenet_synset_map = get_imagenet_id_synset_map()


def synset_label_func(fname):
    words = imagenet_synset_map[Path(fname).parent.name].split(",")
    return ",".join(words[:2])


def make_imagenet_sketch_dls(config):
    device = config["device"]
    path = config.at("data.imagenet_sketch.path")
    valid_pct = config.at("data.imagenet_sketch.valid_pct", 0.2)
    batch_size = config.at("data.imagenet_sketch.batch_size", 64)
    clip_model_name = config.at("clip.model_name", "RN50")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(
        load_clip_preprocess(clip_model_name)
    )
    label_func = (
        simple_label_func
        if config.at("data.imagenet_sketch.labelling.kind") == 'simple'
        else synset_label_func
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


def make_imagenet_dls(config):
    device = config["device"]
    path = config.at("data.imagenet.path")
    batch_size = config.at("data.imagenet.batch_size", 64)
    clip_model_name = config.at("clip.model_name", "RN50")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(
        load_clip_preprocess(clip_model_name)
    )
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=GrandparentSplitter(train_name="train", valid_name="val"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    dls = dblock.dataloaders(path, bs=batch_size, device=device)
    return dls


def load_clip(model_name, prec="fp32"):
    model = clip.load(model_name, device="cpu")[0]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()
    return model


def make_model(class_names, config):
    prompt = config.at("clip.prompt")
    prompted_class_names = [f"{prompt} {c}".strip() for c in class_names]
    clip_model = load_clip(config.at("clip.model_name"), config.at("clip.prec"))
    model = ClipZeroShotClassifier(clip_model, prompted_class_names)
    return model


def run_experiment(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)

    # dataloaders
    print("Creating dataloaders")
    dls = make_imagenet_sketch_dls(config)
    class_names = sorted(list(dls.vocab))
    print(f"#class: {len(class_names)}")

    # model
    print("Creating model")
    model = make_model(class_names, config)

    # evaluation
    print("Creating learner")
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
    )
    print("Evaluating model on train set of target domain")
    clf_summary = evaluate_slmc(learn, dl=dls[0], class_names=class_names)
    accuracy_score = clf_summary.loc["accuracy"][0]
    wandb_run.log(
        {
            "classification-summary": wandb.Table(dataframe=clf_summary.reset_index()),
            "accuracy": accuracy_score,
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg")
    parser.add_argument("--sweep-cfg", required=False)
    args = parser.parse_args()
    main(run_experiment, args)
