#!/bin/bash
HUGGING_FACE_HUB_TOKEN="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"

MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

VOLUME="${HOME}/.cache/huggingface/tgi"
mkdir -p $VOLUME

IMAGE=ghcr.io/huggingface/text-generation-inference:2.0.4

CUDA_VISIBLE_DEVICES=0,1 docker run \
    --gpus all \
    --shm-size 64g \
    -p 8080:80 \
    -v "${VOLUME}":/data \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    $IMAGE \
    --trust-remote-code --model-id $MODEL --dtype bfloat16 --quantize bitsandbytes-nf4 --num-shard 2 