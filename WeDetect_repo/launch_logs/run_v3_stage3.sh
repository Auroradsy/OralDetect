set -e
cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/WeDetect
export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH=.:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TR=/ix/lzhan/siyuan/envs/wedetect_official/bin/torchrun
WD=/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts/oraldetect
S2=$WD/v33stage_s2/epoch_4.pth
[ -f "$S2" ] || { echo "stage2 ckpt missing: $S2"; exit 1; }
rm -rf $WD/v33stage_s3
echo "########## V3 STAGE 3 (4 GPU x batch8 = 32 eff, unfreeze, lr2e-5, 12ep) ##########"
$TR --nproc_per_node=4 --master_port=29564 train.py config/wedetect_v3_3stage_s3.py \
    --launcher pytorch --work-dir $WD/v33stage_s3 \
    --cfg-options load_from=$S2 train_dataloader.batch_size=8 optim_wrapper.optimizer.batch_size_per_gpu=8
echo "ALL V3 STAGE3 DONE"
