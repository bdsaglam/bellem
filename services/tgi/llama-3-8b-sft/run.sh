#!/bin/bash
HUGGING_FACE_HUB_TOKEN="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"

MODEL="bdsaglam/llama-3-8b-jerx-peft-aw7ihmbc"

VOLUME="${HOME}/.cache/huggingface/tgi"
mkdir -p $VOLUME

IMAGE=ghcr.io/huggingface/text-generation-inference:2.0.4

INFERENCE_ARGS="--trust-remote-code --model-id $MODEL --dtype bfloat16 --max-input-tokens 8000 --max-total-tokens 10048"

docker run \
    --gpus all \
    --shm-size 64g \
    -p 8082:80 \
    -v "${VOLUME}":/data \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
    -e CUDA_VISIBLE_DEVICES=1 \
    $IMAGE $INFERENCE_ARGS