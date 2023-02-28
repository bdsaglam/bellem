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


def make_label_func(config):
    return (
        synset_label_func
        if config.at("data.imagenet_sketch.labelling.kind") == "synset"
        else simple_label_func
    )


def make_imagenet_sketch_dls(config):
    clip_model_name = config.at("clip.model_name", "RN50")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(
        load_clip_preprocess(clip_model_name)
    )
    label_func = make_label_func(config)
    valid_pct = config.at("data.imagenet_sketch.valid_pct", 0.2)
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=RandomSplitter(valid_pct=valid_pct),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    device = config["device"]
    path = config.at("data.imagenet_sketch.path")
    batch_size = config.at("data.imagenet_sketch.batch_size", 64)
    dls = dblock.dataloaders(path, bs=batch_size, device=device)
    return dls


def make_imagenet_dls(config):
    clip_model_name = config.at("clip.model_name", "RN50")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(
        load_clip_preprocess(clip_model_name)
    )
    label_func = make_label_func(config)
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=GrandparentSplitter(train_name="train", valid_name="val"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    path = config.at("data.imagenet.path")
    batch_size = config.at("data.imagenet.batch_size", 64)
    device = config["device"]
    dls = dblock.dataloaders(path, bs=batch_size, device=device)
    return dls


def make_pdls(source_dls, target_dls):
    return (
        McdDataLoader(source_dls.train, target_dls.train),
        McdDataLoader(source_dls.valid, target_dls.valid),
    )


def make_dls(config):
    imagenette_dls = make_imagenet_dls(config)
    imagenette_sketch_dls = make_imagenet_sketch_dls(config)
    train_pdl, valid_pdl = make_pdls(imagenette_dls, imagenette_sketch_dls)
    dls = DataLoaders(train_pdl, valid_pdl, device=config["device"])
    dls.n_inp = 2
    return dls


class CoCoopClassifier(nn.Module):
    def __init__(self, clip_model, tokenizer, class_names, **kwargs):
        super().__init__()
        self.text_encoder = PromptLearningTextEncoder(
            clip_model, tokenizer, class_names, **kwargs
        )
        self.head = ClipClassificationHead(clip_model)

    def forward(self, image_features):
        text_features = self.text_encoder()
        logits = self.head(image_features, text_features)
        return logits


def load_clip(model_name, prec="fp32"):
    model = clip.load(model_name, device="cpu")[0]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()
    return model


def make_cocoop_feature_extractor(clip_model, trainable=False):
    model = ClipVisualEncoder(clip_model)
    for param in model.parameters():
        param.requires_grad_(trainable)
    return model


def make_cocoop_classifier(clip_model, class_names, **kwargs):
    tokenizer = SimpleTokenizer()
    return CoCoopClassifier(clip_model, tokenizer, class_names, **kwargs)


def make_model(class_names, config):
    clip_model = load_clip(**config.get("clip"))
    model = McdModel(
        make_cocoop_feature_extractor(clip_model, trainable=False),
        make_cocoop_classifier(clip_model, class_names, **config.get("cocoop")),
        make_cocoop_classifier(clip_model, class_names, **config.get("cocoop")),
    )
    return model


def evaluate_ensemble(learn, dls):
    ensemble_model = EnsembleMcdModel.from_mcd_model(learn.model)
    elearn = Learner(
        dls,
        ensemble_model,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
    )
    return evaluate_slmc(elearn, dls=dls, show=False)


def run_experiment(wandb_run):
    config = NestedDict.from_flat_dict(wandb_run.config)
    seed = config.get("seed")
    if seed is not None:
        set_seed(seed)

    # dataloaders
    print("Creating dataloaders")
    imagenette_dls = make_imagenet_dls(config)
    imagenette_sketch_dls = make_imagenet_sketch_dls(config)
    train_pdl, valid_pdl = make_pdls(imagenette_dls, imagenette_sketch_dls)
    dls = DataLoaders(train_pdl, valid_pdl, device=config["device"])
    dls.n_inp = 2
    class_names = sorted(list(dls.vocab))
    print(f"#class: {len(class_names)}")

    # model
    print("Creating model")
    model = make_model(class_names, config)

    # training
    cbs = [SaveModelCallback(), WandbCallback()]
    if config.at("train.early_stop"):
        cbs.append(
            EarlyStoppingCallback(patience=config.at("train.early_stop.patience"))
        )

    print("Creating learner")
    learn = mcd_learner(
        dls,
        model,
        cbs=cbs,
    )
    print(f"Training on {config.get('device')}")
    learn.fit(config.at("train.n_epoch"), lr=config.at("train.lr"))

    # evaluation
    print("Evaluating model on validation set of target domain")
    clf_summary = evaluate_ensemble(learn, imagenette_sketch_dls)
    wandb_run.log(
        {
            "classification-summary": wandb.Table(dataframe=clf_summary.reset_index()),
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg")
    parser.add_argument("--sweep-cfg", required=False)
    args = parser.parse_args()
    main(run_experiment, args)
