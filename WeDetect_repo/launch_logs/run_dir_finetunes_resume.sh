#!/bin/bash
# TODO / run-later: resume the two paused direct finetunes to epoch 12.
#   base-dir     : epoch_9  -> 12  (config wedetect_dental_oraldetect.py)   -> GPU 4
#   oralclip-dir : epoch_10 -> 12  (config wedetect_oraldetect_v11dt.py)    -> GPU 5
# Both use --resume (auto-picks the latest epoch ckpt). Change CUDA_VISIBLE_DEVICES if needed.
cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/WeDetect
export PYTHONPATH=.:$PYTHONPATH
PY=/ix/lzhan/siyuan/envs/wedetect_official/bin/python
LOGDIR=/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/launch_logs

CUDA_VISIBLE_DEVICES=4 setsid $PY train.py config/wedetect_dental_oraldetect.py \
    --work-dir work_dirs/oraldetect_basedir --resume >> $LOGDIR/oraldetect_basedir.log 2>&1 &
echo "base-dir resume (GPU4) PID $!"
CUDA_VISIBLE_DEVICES=5 setsid $PY train.py config/wedetect_oraldetect_v11dt.py \
    --work-dir work_dirs/oraldetect_oralclipdir --resume >> $LOGDIR/oraldetect_oralclipdir.log 2>&1 &
echo "oralclip-dir resume (GPU5) PID $!"
