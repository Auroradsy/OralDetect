#!/bin/bash
# Resume 1024 v11dt from STAGE 2 (stage 1 already completed: epoch_8.pth). 4 GPUs.
set -e
cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/WeDetect
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=.:$PYTHONPATH
TR=/ix/lzhan/siyuan/envs/wedetect_official/bin/torchrun
WD=/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts/oraldetect
S1=$WD/oraldetect_v11dt_1024_s1/epoch_8.pth
[ -f "$S1" ] || { echo "stage1 ckpt missing: $S1"; exit 1; }

echo "########## STAGE 2 @1024 (freeze, consolidate, lr 3e-5, 4ep) ##########"
rm -rf $WD/oraldetect_v11dt_1024_s2
$TR --nproc_per_node=4 --master_port=29592 train.py config/wedetect_oraldetect_v11dt_1024_s2.py \
    --launcher pytorch --work-dir $WD/oraldetect_v11dt_1024_s2 --cfg-options load_from=$S1
S2=$WD/oraldetect_v11dt_1024_s2/epoch_4.pth
[ -f "$S2" ] || { echo "STAGE2 ckpt missing: $S2"; exit 1; }

echo "########## STAGE 3 @1024 (unfreeze all, e2e, lr 2e-5, 12ep) ##########"
rm -rf $WD/oraldetect_v11dt_1024_s3
$TR --nproc_per_node=4 --master_port=29593 train.py config/wedetect_oraldetect_v11dt_1024_s3.py \
    --launcher pytorch --work-dir $WD/oraldetect_v11dt_1024_s3 --cfg-options load_from=$S2
echo "STAGES 2-3 DONE (1024)"
