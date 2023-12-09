model=NousResearch/llama-2-7b-chat-hf

volume="${HOME}/hf-tgi/data"
mkdir $volume

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.2 --model-id $model