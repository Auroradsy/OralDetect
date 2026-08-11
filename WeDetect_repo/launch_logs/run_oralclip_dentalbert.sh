cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/WeDetect
export CUDA_VISIBLE_DEVICES=2,3,4,5
export PYTHONPATH=.:$PYTHONPATH
/ix/lzhan/siyuan/envs/wedetect_official/bin/torchrun \
  --nproc_per_node=4 --master_port=29523 \
  train.py config/wedetect_dental_mode1_oralclip_dentalbert.py --launcher pytorch \
  --work-dir work_dirs/wedetect_dental_mode1_oralclip_dentalbert
