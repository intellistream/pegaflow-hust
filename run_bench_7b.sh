#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
cd /workspace/HUST/pegaflow-hust
pkill -f "pegaflow-server" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
sleep 2
python run_npu_benchmark.py 4 2>&1
