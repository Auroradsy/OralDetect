set -e
cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/WeDetect
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=.:$PYTHONPATH
TR=/ix/lzhan/siyuan/envs/wedetect_official/bin/torchrun
WD=/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts/oraldetect
BATCH="train_dataloader.batch_size=16 optim_wrapper.optimizer.batch_size_per_gpu=16"
S1=$WD/v33stage_s1/epoch_8.pth   # stage 1 already complete
[ -f "$S1" ] || { echo "stage1 ckpt missing: $S1"; exit 1; }

echo "########## V3 STAGE 2 (RESUME from latest in work_dir, lr3e-5, ->4ep) ##########"
$TR --nproc_per_node=2 --master_port=29562 train.py config/wedetect_v3_3stage_s2.py \
    --launcher pytorch --work-dir $WD/v33stage_s2 --resume --cfg-options load_from=$S1 $BATCH
S2=$WD/v33stage_s2/epoch_4.pth
[ -f "$S2" ] || { echo "V3-STAGE2 ckpt missing: $S2"; exit 1; }

echo "########## V3 STAGE 3 (unfreeze all, e2e, lr2e-5, 12ep) ##########"
$TR --nproc_per_node=2 --master_port=29563 train.py config/wedetect_v3_3stage_s3.py \
    --launcher pytorch --work-dir $WD/v33stage_s3 --cfg-options load_from=$S2 $BATCH

echo "ALL V3 (resumed) 3 STAGES DONE"
