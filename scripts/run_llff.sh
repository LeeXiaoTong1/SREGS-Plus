#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh

source activate FS

SCENE="trex"

python train_llff.py \
  -s dataset/llff/${SCENE} \
  -m output/llff/${SCENE}_qwen_online \
  --eval \
  --n_views 3 \
  --qwen_online

conda deactivate