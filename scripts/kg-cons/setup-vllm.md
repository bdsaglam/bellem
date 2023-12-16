# Create virtual env
conda create -n vllm python=3.9 -y
conda activate vllm

# Install vLLM with CUDA 12.1
<!-- conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit cuda-nvcc
conda install pytorch torchvision torchaudio pytorch-cuda=12.1.1 -c pytorch -c nvidia -->
pip install vllm

# Install vLLM with CUDA 11.8
export VLLM_VERSION=0.2.4
export PYTHON_VERSION=39
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl
pip uninstall torch -y
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118

# Start OpenAI API compatible inference server

python -m vllm.entrypoints.openai.api_server --trust-remote-code --model NousResearch/Llama-2-7b-chat-hf