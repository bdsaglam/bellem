#!/bin/bash

#SBATCH -J jobname
#SBATCH -A basaglam
#SBATCH -p akya-cuda
#SBATCH --time=20:00
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10

set -euo pipefail
set -x

# beginning
date --iso-8601=seconds

# check GPUs
nvidia-smi

# activate virtual env
eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)" &> /dev/null
conda activate fastexp

# train
python experiment.py --cfg ./config.json

# end
date --iso-8601=seconds
exit