#!/bin/bash
#SBATCH --job-name=od_eval
#SBATCH --cluster=gpu --partition=a100 --account=lzhan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --mem=180g --time=04:00:00
#SBATCH --output=/ix/lzhan/siyuan/exps/OralDetect_Family/od_eval.%j.log
#
#   sbatch OralDetect/launch_bash/eval_oraldetect.sh
#
# Everything configurable is in eval_oraldetect.yaml; everything procedural is in run_eval.py
# (path checks, the checkpoint key diff, the scoring loop over benches, the summary table).
# GPU count lives in --gres above and in the yaml's eval.gpus; run_eval.py asserts they agree.
set -euo pipefail

REPO=/ihome/lzhan/sid51/projects/VLM/OralDetect-Family
YAML=$REPO/OralDetect/launch_bash/eval_oraldetect.yaml
NPROC=4
PORT=29812                                             # differs from finetune's 29811

module load micromamba/2.5.0
export MAMBA_ROOT_PREFIX=/ix/lzhan/siyuan/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate /ix/lzhan/siyuan/envs/wedetect_official

export PYTHONUNBUFFERED=1
export HF_HOME=/ix/lzhan/siyuan/.cache/huggingface     # keep caches off $HOME (75 G, cannot grow)
export TORCH_HOME=/ix/lzhan/siyuan/baselines/torch_cache

torchrun --nproc_per_node=$NPROC --master_port=$PORT "$REPO/run_eval.py" --config "$YAML"
