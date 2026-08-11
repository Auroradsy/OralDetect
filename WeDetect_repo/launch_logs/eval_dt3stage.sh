cd /ihome/lzhan/sid51/projects/VLM/OralDetect-Family/WeDetect_repro/WeDetect
export PYTHONPATH=.:$PYTHONPATH
CUDA_VISIBLE_DEVICES=2,3,4,5 /ix/lzhan/siyuan/envs/wedetect_official/bin/torchrun   --nproc_per_node=4 --master_port=29534   test.py config/wedetect_dt_3stage_s3.py "work_dirs/dt3stage_s3/best_coco_bbox_mAP_epoch_12.pth" --launcher pytorch   --cfg-options test_evaluator.outfile_prefix=work_dirs/dt3stage_s3/preds_dt3stage
echo "DT3STAGE EVAL DONE"
