#!/bin/bash
# Reorg direct end-to-end finetunes (12 ep, no freeze), 4 GPUs, for init comparison vs v11dt-3stage:
#   base-dir : WeDetect base init + XLM-R text        (config wedetect_dental_reorg.py)
#   v11-dir  : OralCLIP-v11 vision + DentalBERT text   (config wedetect_reorg_v11dt.py)
set -e
cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/WeDetect
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=.:$PYTHONPATH
TR=/ix/lzhan/siyuan/envs/wedetect_official/bin/torchrun
WD=/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts/oraldetect
rm -rf $WD/reorg_basedir $WD/reorg_v11dir

echo "########## base-dir : direct e2e finetune (base init + XLM-R, 12ep) ##########"
$TR --nproc_per_node=4 --master_port=29581 train.py config/wedetect_dental_reorg.py \
    --launcher pytorch --work-dir $WD/reorg_basedir

echo "########## v11-dir : direct e2e finetune (OralCLIP-v11 + DentalBERT, 12ep) ##########"
$TR --nproc_per_node=4 --master_port=29582 train.py config/wedetect_reorg_v11dt.py \
    --launcher pytorch --work-dir $WD/reorg_v11dir

echo "BOTH DIR FINETUNES DONE"
