#!/bin/bash

model='bdsaglam/llama-2-7b-chat-jerx-mt-ss-peft-2024-02-04T00-14-15'

VOLUME="${HOME}/.cache/huggingface/tgi"
mkdir -p $VOLUME

docker run --gpus all --shm-size 1g \
    -p 8080:80 \
    -v "${VOLUME}":/data \
    ghcr.io/huggingface/text-generation-inference:1.3 \
    --trust-remote-code \
    --model-id $model --dtype float16