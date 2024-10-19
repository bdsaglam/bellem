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
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
from fastai.callback.wandb import *
from fastai.data.all import *
from fastai.torch_core import set_seed
from fastai.vision.all import *
from fastcore.basics import ifnone, store_attr
from torchvision import transforms

import wandb
from bellek.ml.clip import *
from bellek.ml.coop import *
from bellek.ml.data import *
from bellek.ml.evaluation import evaluate_slmc
from bellek.ml.experiment import *
from bellek.ml.mcd import *
from bellek.ml.vision import *
from bellek.utils import *
from bellek.ds import *


def make_office_home_dls(config, domain):
    valid_pct = config.at("data.office_home.valid_pct", 0.3)
    batch_size = config.at("data.office_home.batch_size", 64)
    clip_model_name = config.at("clip.model_name")
    item_tfms, batch_tfms = make_tfms_from_clip_preprocess(load_clip_preprocess(clip_model_name))
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=lambda p: Path(p).parent.name.lower().replace("_", " "),
        splitter=RandomSplitter(valid_pct=valid_pct),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    path = Path(config.at("data.office_home.path"))
    dls = dblock.dataloaders(path / domain, bs=batch_size)
    return dls


def make_pdls(source_dls, target_dls):
    return (
        McdDataLoader(source_dls.train, target_dls.train),
        McdDataLoader(source_dls.valid, target_dls.valid),
    )


class CoopClassifier(nn.Module):
    def __init__(self, clip_model, class_names, **kwargs):
        super().__init__()
        self.text_encoder = PromptLearningTextEncoder(clip_model, SimpleTokenizer(), class_names, **kwargs )
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


def make_coop_feature_extractor(clip_model, trainable=False):
    model = ClipVisualEncoder(clip_model)
    for param in model.parameters():
        param.requires_grad_(trainable)
    return model


def make_model(class_names, config):
    clip_model = load_clip(**config.get("clip"))
    model = McdModel(
        make_coop_feature_extractor(clip_model, trainable=False),
        CoopClassifier(clip_model, class_names, **config.get("coop")),
        CoopClassifier(clip_model, class_names, **config.get("coop")),
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
    source_dls = make_office_home_dls(config, 'Real World')
    target_dls = make_office_home_dls(config, 'Clipart')
    train_pdl, valid_pdl = make_pdls(source_dls, target_dls)
    dls = DataLoaders(train_pdl, valid_pdl, device=config["device"])
    dls.n_inp = 2
    class_names = sorted(list(dls.vocab))
    print(f"#class: {len(class_names)}")

    # model
    print("Creating model")
    model = make_model(class_names, config)

    # training
    cbs = [WandbCallback(log_preds=False)]
    if config.at("train.early_stop"):
        cbs.append(EarlyStoppingCallback(patience=config.at("train.early_stop.patience")))

    print("Creating learner")
    learn = mcd_learner(dls, model, cbs=cbs)
    print(f"Training on {config.get('device')}")
    learn.fit(config.at("train.n_epoch"), lr=config.at("train.lr"))

    # evaluation
    print("Evaluating model on validation set of target domain")
    clf_summary = evaluate_ensemble(learn, target_dls)
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
