#!/bin/bash
#SBATCH -J merge10layers-sweep
#SBATCH -N1 --gres=gpu:H100:1 --mem=32G
#SBATCH -t 30
#SBATCH -q coc-ice
#SBATCH -o output.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vvobbilisetty6@gatech.edu
cd /home/hice1/vvobbilisetty6/scratch/mergekit/mergekit
module load anaconda3
conda activate vllm310

vllm serve with_special_tokens --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.6 --enforce-eager  &
VLLM_PID=$!

sleep 500


for reqr in 256 512 1024 2048 3500 4096; do
    vllm bench serve \
        --model with_special_tokens \
        --port 8000 \
        --dataset-name custom \
        --dataset-path custom_prompts.jsonl \
        --max-concurrency 512 \
        --request-rate $reqr \
        --num-prompts 5000 \
        --save-result \
        --result-dir vllm_results_10merge_new \
        --metric-percentiles 95,99 
done

python visualize_results.py

kill $VLLM_PID