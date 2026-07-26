#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
cd /workspace/HUST/pegaflow-hust
python run_npu_benchmark.py 2