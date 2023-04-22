#!/bin/bash

set -o pipefail

## 1. Download the data

# get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
if [ ! -f ILSVRC2012_img_val.tar ]
then
    wget --no-check-certificate https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
fi

## 3. Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash