cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family
export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=OralCLIP/ckpts/clip_checkpoints_v9/epoch_12.pt
/ix/lzhan/siyuan/envs/wedetect_official/bin/torchrun --nproc_per_node=2 --master_port=29529 \
  OralCLIP/train_clip_v9.py --output-dir OralCLIP/ckpts/clip_checkpoints_v9 \
  --epochs 30 --batch-size 128 --lr 5e-4 --warmup-epochs 2.0 \
  --lambda-mod 0.1 --text-proj mlp --max-text-len 256 --workers 8 --resume "$CKPT"
