#!/bin/bash
MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

VOLUME="${HOME}/.cache/huggingface/tgi"
mkdir -p $VOLUME

IMAGE=bdsaglam/text-generation-inference:latest

docker run \
    --gpus "2,3" \
    --shm-size 1g \
    -p 8080:80 \
    -v "${VOLUME}":/data \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
    $IMAGE \
    --trust-remote-code --model-id $MODEL --dtype bfloat16 --quantize bitsandbytes-nf4 --num-shard 2 