#!/bin/bash

set -o pipefail

cd $1

## 1. Download the data

# get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
if [ ! -f ILSVRC2012_img_train.tar ]
then
    wget --no-check-certificate https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
fi

## 2. Extract the training data:
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

## 3. Delete corrupted image
# there is one png under JPEG name. some readers fail on this image so need to remove it
# this line is commented by default to avoid such unexpected behaviour
rm train/n04266014/n04266014_10835.JPEG
