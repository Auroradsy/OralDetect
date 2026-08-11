#!/bin/bash
#SBATCH --job-name=od_ft
#SBATCH --cluster=gpu --partition=a100 --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --mem=180g --time=08:00:00
#SBATCH --output=/ix/lzhan/siyuan/exps/OralDetect_Family/od_ft.%j.log
#
#   sbatch OralDetect/launch_bash/finetune_oraldetect.sh
#
# Everything configurable is in finetune_oraldetect.yaml; everything procedural is in
# run_finetune.py (path checks, the resolved-config summary, the load_from key diff, auto-resume).
# GPU count lives in --gres above and in the yaml's train.gpus; run_finetune.py asserts they agree.
set -euo pipefail

REPO=/ihome/lzhan/sid51/projects/VLM/OralDetect-Family
YAML=$REPO/OralDetect/launch_bash/finetune_oraldetect.yaml
NPROC=4
PORT=29811

module load micromamba/2.5.0
export MAMBA_ROOT_PREFIX=/ix/lzhan/siyuan/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate /ix/lzhan/siyuan/envs/wedetect_official

export PYTHONUNBUFFERED=1
export HF_HOME=/ix/lzhan/siyuan/.cache/huggingface     # keep caches off $HOME (75 G, cannot grow)
export TORCH_HOME=/ix/lzhan/siyuan/baselines/torch_cache

torchrun --nproc_per_node=$NPROC --master_port=$PORT "$REPO/run_finetune.py" --config "$YAML"
