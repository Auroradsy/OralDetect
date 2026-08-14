"""Finetune OralDetect from our own trained detector -- SELF-CONTAINED, no `_base_`.

This is a FLATTENED config: mmengine's `Config.dump()` resolved what used to be a five-level
inheritance chain (finetune -> v14_base -> base_1024 -> wedetect_base -> default_runtime) into one
file, so this directory now holds exactly one config and nothing here depends on anything else.
Verified equivalent to the chain it replaced: 70/70 top-level keys equal, and the models built from
each have identical state_dicts (1126 params, 0 key or shape differences).

Paths, vocabulary and hyper-parameters are OVERWRITTEN AT RUNTIME by `run_finetune.py` from
`OralDetect/launch_bash/finetune_oraldetect.yaml`. The values baked in below are the defaults --
edit the yaml, not this file.

WHAT IT STARTS FROM. `load_from` is `v14_calib_v2/stageC/best_coco_macro4_mAP_epoch_6.pth`: the
model the paper reports, selected on MACRO-4. NOT `wedetect_base.pth` and NOT any `*_init.pth` --
those are pretrains that still have to learn detection. This one is already a converged dental
detector, which is why the schedule below is short and slow.

THE ARCHITECTURE HERE MUST MATCH THAT CHECKPOINT. `load_from` in mmengine is forgiving: keys that
do not match are dropped with a log line and training proceeds, so a drifted architecture gives you
a partly random model that trains happily and is quietly worse. `run_finetune.py` guards this --
it builds the model, diffs the state_dict against the checkpoint, and refuses to start on any
missing / unexpected / mis-shaped key (`strict_load: false` in the yaml to override). The three
things that must not change: the `DentalBertLanguageBackbone` text tower, `ModalityCalibration`,
and 1024x1024.

SWAPPING IN A DIFFERENT DATASET -- two things bite:

  * Vocabulary size is SHAPE-SAFE up to 256. YOLOWorldHeadModule sizes its classifier as
    `cls_out_channels = max(in_channels[0], num_classes)` = `max(256, n)`, and the class prototypes
    come from the text tower rather than a learned per-class matrix. Past 256 the cls_preds convs
    change shape and load_from drops them -- that is a different experiment, not a finetune.
    `run_finetune.py` refuses past 256.
  * `PerModalityCocoMetric` groups by an explicit `modality` key on every image record. A dataset
    built outside this project will not have one and the metric raises. Set `data.evaluator: coco`
    in the yaml. You then get plain COCO mAP, which is a CLASS-macro over wildly unbalanced class
    slots -- read CLAUDE.md section 3 before comparing that number to anything.

SCHEDULE -- deliberately not the 12-epoch recipe, which starts from a pretrain and has to learn
detection. This starts from a converged detector, so the job is to adapt, not to train:
  * lr 5e-6, a quarter of the from-scratch 2e-5. Stage C already ran the tail of its schedule at
    2e-6; coming back up to 2e-5 would undo it.
  * 6 epochs, mosaic OFF THROUGHOUT (`train_pipeline_stage2` is the train pipeline from epoch 0,
    and `custom_hooks` is empty so no switch hook fires). Mosaic composites four images and teaches
    robustness to context that never appears at test time -- worth it when learning detection from
    scratch, counterproductive for a short adaptation.
  * warmup 200 iters, not 1000: 1000 is a large slice of a 6-epoch run, and nothing here is
    randomly initialised, so there is nothing to ease in.
  * val every epoch -- a 6-epoch run has no room for a 2-epoch blind spot.
  * save_best on MACRO-4, not MACRO-5: histology's val split is 35 images and swings +-0.13 between
    adjacent epochs, so letting it vote means 35 images pick the model. See CLAUDE.md section 3.

The archived multi-file lineage (v14 stages, the abl640 arms, tokenlen, openset, and every earlier
recipe) is in `OralDetect/_outdate_files/WeDetect/config/`.
"""

CLASS_NAMES = '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/class_names_oraldetect.json'
CLASS_TEXT = '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/class_texts_oraldetect.json'
DATAS = '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas'
DATA_ROOT = '/ix/lzhan/siyuan/datasets/processed_datas/OralDetect_Family/OralDetect_data_by_modality/'
DENTALBERT = '/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts/oralbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_dental_mlm/final'
TEST_ANN = '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_val.json'
TRAIN_ANN = '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_train.json'
affine_scale = 0.5
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
backend_args = None
base_lr = 5e-06
close_mosaic_epochs = 4
custom_hooks = []
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'wedetect',
    ])
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/macro4_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmdet'
dist_cfg = dict(backend='nccl', timeout=10800)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
img_scale = (
    1024,
    1024,
)
load_from = '/ix/lzhan/siyuan/exps/OralDetect_Family/v14_calib_v2/stageC/best_coco_macro4_mAP_epoch_6.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
max_epochs = 6
metainfo = dict(
    classes=(
        'abnormal oral epithelial cell',
        'alveolar bone loss',
        'anterior teeth with fenestration or dehiscence',
        'anterior teeth without fenestration or dehiscence',
        'canine',
        'central incisor',
        'craniofacial or oral anomaly',
        'dental abrasion',
        'dental caries',
        'dental crown restoration',
        'dental filling',
        'dental implant',
        'dental opacity',
        'dental plaque',
        'dental restoration',
        'dental restoration (filling or crown)',
        'dividing oral cell',
        'first molar',
        'first premolar',
        'foreign object or debris',
        'hard recognized oral cell',
        'impacted tooth',
        'intraoral appliance',
        'lateral incisor',
        'lightly abnormal oral cell',
        'malignant oral cell',
        'mandibular canal',
        'maxillary sinus',
        'missing or residual root',
        'missing teeth',
        'normal',
        'normal oral cell',
        'oral blood cell',
        'orthodontic bracket',
        'periapical lesion',
        'periodontal pocket',
        'primary endodontic lesion',
        'primary endodontic with secondary periodontal lesion',
        'primary periodontal lesion',
        'primary periodontal with secondary endodontic lesion',
        'quadrant 1 (upper right)',
        'quadrant 2 (upper left)',
        'quadrant 3 (lower left)',
        'quadrant 4 (lower right)',
        'reactive oral cell',
        'retained root',
        'root canal treatment',
        'second molar',
        'second premolar',
        'severe gingivitis',
        'suspicious malignant oral cell',
        'tooth 11 (upper right central incisor)',
        'tooth 12 (upper right lateral incisor)',
        'tooth 13 (upper right canine)',
        'tooth 14 (upper right first premolar)',
        'tooth 15 (upper right second premolar)',
        'tooth 16 (upper right first molar)',
        'tooth 17 (upper right second molar)',
        'tooth 18 (upper right third molar)',
        'tooth 21 (upper left central incisor)',
        'tooth 22 (upper left lateral incisor)',
        'tooth 23 (upper left canine)',
        'tooth 24 (upper left first premolar)',
        'tooth 25 (upper left second premolar)',
        'tooth 26 (upper left first molar)',
        'tooth 27 (upper left second molar)',
        'tooth 28 (upper left third molar)',
        'tooth 31 (lower left central incisor)',
        'tooth 32 (lower left lateral incisor)',
        'tooth 33 (lower left canine)',
        'tooth 34 (lower left first premolar)',
        'tooth 35 (lower left second premolar)',
        'tooth 36 (lower left first molar)',
        'tooth 37 (lower left second molar)',
        'tooth 38 (lower left third molar)',
        'tooth 41 (lower right central incisor)',
        'tooth 42 (lower right lateral incisor)',
        'tooth 43 (lower right canine)',
        'tooth 44 (lower right first premolar)',
        'tooth 45 (lower right second premolar)',
        'tooth 46 (lower right first molar)',
        'tooth 47 (lower right second molar)',
        'tooth 48 (lower right third molar)',
        'tooth erosion',
        'tooth malformation',
        'true combined endo-perio lesion',
        'tumor',
    ))
mixup_prob = 0.15
model = dict(
    backbone=dict(
        image_model=dict(
            frozen_modules=[],
            model_name='base',
            type='ConvNextVisionBackbone'),
        text_model=dict(
            frozen_modules=[],
            model_name=
            '/ix/lzhan/siyuan/exps/OralDetect_Family/our_ckpts/oralbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_dental_mlm/final',
            model_size='base',
            type='DentalBertLanguageBackbone'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='WeDetectDistancePointBBoxCoder'),
        head_module=dict(
            embed_dims=768,
            in_channels=[
                256,
                512,
                1024,
            ],
            model_size='base',
            num_classes=87,
            type='YOLOWorldHeadModule',
            use_bn_head=True),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='mmyoloIoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375, reduction='mean', type='DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        type='YOLOWorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOWDetDataPreprocessor'),
    mm_neck=False,
    modality_calib=dict(
        dropout=0.0,
        embed_dims=768,
        num_heads=8,
        num_modality_tokens=4,
        type='ModalityCalibration',
        vision_dims=1024,
        vision_index=-1),
    neck=dict(model_size='base', scale_factor=1.0, type='CSPRepBiFPANNeck'),
    num_test_classes=87,
    num_train_classes=87,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=87,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLOWorldDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            1024,
            1024,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(
        border=(
            -512,
            -512,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='WeDetectRandomAffine'),
]
neck_embed_channels = [
    128,
    256,
    512,
]
neck_num_heads = [
    4,
    8,
    16,
]
num_classes = 87
num_training_classes = 87
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOWv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=4, lr=5e-06, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(custom_keys=dict(logit_scale=dict(weight_decay=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=200,
        end_factor=1.0,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=6,
        end_factor=0.01,
        start_factor=1.0,
        type='LinearLR'),
]
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
resume = False
save_epoch_intervals = 1
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path=
        '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/class_texts_oraldetect.json',
        dataset=dict(
            ann_file=
            '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img=''),
            data_root=
            '/ix/lzhan/siyuan/datasets/processed_datas/OralDetect_Family/OralDetect_data_by_modality/',
            metainfo=dict(
                classes=(
                    'abnormal oral epithelial cell',
                    'alveolar bone loss',
                    'anterior teeth with fenestration or dehiscence',
                    'anterior teeth without fenestration or dehiscence',
                    'canine',
                    'central incisor',
                    'craniofacial or oral anomaly',
                    'dental abrasion',
                    'dental caries',
                    'dental crown restoration',
                    'dental filling',
                    'dental implant',
                    'dental opacity',
                    'dental plaque',
                    'dental restoration',
                    'dental restoration (filling or crown)',
                    'dividing oral cell',
                    'first molar',
                    'first premolar',
                    'foreign object or debris',
                    'hard recognized oral cell',
                    'impacted tooth',
                    'intraoral appliance',
                    'lateral incisor',
                    'lightly abnormal oral cell',
                    'malignant oral cell',
                    'mandibular canal',
                    'maxillary sinus',
                    'missing or residual root',
                    'missing teeth',
                    'normal',
                    'normal oral cell',
                    'oral blood cell',
                    'orthodontic bracket',
                    'periapical lesion',
                    'periodontal pocket',
                    'primary endodontic lesion',
                    'primary endodontic with secondary periodontal lesion',
                    'primary periodontal lesion',
                    'primary periodontal with secondary endodontic lesion',
                    'quadrant 1 (upper right)',
                    'quadrant 2 (upper left)',
                    'quadrant 3 (lower left)',
                    'quadrant 4 (lower right)',
                    'reactive oral cell',
                    'retained root',
                    'root canal treatment',
                    'second molar',
                    'second premolar',
                    'severe gingivitis',
                    'suspicious malignant oral cell',
                    'tooth 11 (upper right central incisor)',
                    'tooth 12 (upper right lateral incisor)',
                    'tooth 13 (upper right canine)',
                    'tooth 14 (upper right first premolar)',
                    'tooth 15 (upper right second premolar)',
                    'tooth 16 (upper right first molar)',
                    'tooth 17 (upper right second molar)',
                    'tooth 18 (upper right third molar)',
                    'tooth 21 (upper left central incisor)',
                    'tooth 22 (upper left lateral incisor)',
                    'tooth 23 (upper left canine)',
                    'tooth 24 (upper left first premolar)',
                    'tooth 25 (upper left second premolar)',
                    'tooth 26 (upper left first molar)',
                    'tooth 27 (upper left second molar)',
                    'tooth 28 (upper left third molar)',
                    'tooth 31 (lower left central incisor)',
                    'tooth 32 (lower left lateral incisor)',
                    'tooth 33 (lower left canine)',
                    'tooth 34 (lower left first premolar)',
                    'tooth 35 (lower left second premolar)',
                    'tooth 36 (lower left first molar)',
                    'tooth 37 (lower left second molar)',
                    'tooth 38 (lower left third molar)',
                    'tooth 41 (lower right central incisor)',
                    'tooth 42 (lower right lateral incisor)',
                    'tooth 43 (lower right canine)',
                    'tooth 44 (lower right first premolar)',
                    'tooth 45 (lower right second premolar)',
                    'tooth 46 (lower right first molar)',
                    'tooth 47 (lower right second molar)',
                    'tooth 48 (lower right third molar)',
                    'tooth erosion',
                    'tooth malformation',
                    'true combined endo-perio lesion',
                    'tumor',
                )),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1024,
                1024,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1024,
                    1024,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_val.json',
    metric='bbox',
    type='PerModalityCocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        1024,
        1024,
    ), type='WeDetectKeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            1024,
            1024,
        ),
        type='WeDetectLetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='PackDetInputs'),
]
text_channels = 768
text_transform = [
    dict(
        max_num_samples=87,
        num_neg_samples=(
            87,
            87,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_batch_size_per_gpu = 4
train_cfg = dict(
    dynamic_intervals=None,
    max_epochs=6,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        class_text_path=
        '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/class_texts_oraldetect.json',
        dataset=dict(
            ann_file=
            '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_train.json',
            data_prefix=dict(img=''),
            data_root=
            '/ix/lzhan/siyuan/datasets/processed_datas/OralDetect_Family/OralDetect_data_by_modality/',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
                    'abnormal oral epithelial cell',
                    'alveolar bone loss',
                    'anterior teeth with fenestration or dehiscence',
                    'anterior teeth without fenestration or dehiscence',
                    'canine',
                    'central incisor',
                    'craniofacial or oral anomaly',
                    'dental abrasion',
                    'dental caries',
                    'dental crown restoration',
                    'dental filling',
                    'dental implant',
                    'dental opacity',
                    'dental plaque',
                    'dental restoration',
                    'dental restoration (filling or crown)',
                    'dividing oral cell',
                    'first molar',
                    'first premolar',
                    'foreign object or debris',
                    'hard recognized oral cell',
                    'impacted tooth',
                    'intraoral appliance',
                    'lateral incisor',
                    'lightly abnormal oral cell',
                    'malignant oral cell',
                    'mandibular canal',
                    'maxillary sinus',
                    'missing or residual root',
                    'missing teeth',
                    'normal',
                    'normal oral cell',
                    'oral blood cell',
                    'orthodontic bracket',
                    'periapical lesion',
                    'periodontal pocket',
                    'primary endodontic lesion',
                    'primary endodontic with secondary periodontal lesion',
                    'primary periodontal lesion',
                    'primary periodontal with secondary endodontic lesion',
                    'quadrant 1 (upper right)',
                    'quadrant 2 (upper left)',
                    'quadrant 3 (lower left)',
                    'quadrant 4 (lower right)',
                    'reactive oral cell',
                    'retained root',
                    'root canal treatment',
                    'second molar',
                    'second premolar',
                    'severe gingivitis',
                    'suspicious malignant oral cell',
                    'tooth 11 (upper right central incisor)',
                    'tooth 12 (upper right lateral incisor)',
                    'tooth 13 (upper right canine)',
                    'tooth 14 (upper right first premolar)',
                    'tooth 15 (upper right second premolar)',
                    'tooth 16 (upper right first molar)',
                    'tooth 17 (upper right second molar)',
                    'tooth 18 (upper right third molar)',
                    'tooth 21 (upper left central incisor)',
                    'tooth 22 (upper left lateral incisor)',
                    'tooth 23 (upper left canine)',
                    'tooth 24 (upper left first premolar)',
                    'tooth 25 (upper left second premolar)',
                    'tooth 26 (upper left first molar)',
                    'tooth 27 (upper left second molar)',
                    'tooth 28 (upper left third molar)',
                    'tooth 31 (lower left central incisor)',
                    'tooth 32 (lower left lateral incisor)',
                    'tooth 33 (lower left canine)',
                    'tooth 34 (lower left first premolar)',
                    'tooth 35 (lower left second premolar)',
                    'tooth 36 (lower left first molar)',
                    'tooth 37 (lower left second molar)',
                    'tooth 38 (lower left third molar)',
                    'tooth 41 (lower right central incisor)',
                    'tooth 42 (lower right lateral incisor)',
                    'tooth 43 (lower right canine)',
                    'tooth 44 (lower right first premolar)',
                    'tooth 45 (lower right second premolar)',
                    'tooth 46 (lower right first molar)',
                    'tooth 47 (lower right second molar)',
                    'tooth 48 (lower right third molar)',
                    'tooth erosion',
                    'tooth malformation',
                    'true combined endo-perio lesion',
                    'tumor',
                )),
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                1024,
                1024,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    1024,
                    1024,
                ),
                type='WeDetectLetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='WeDetectRandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='WeDetectHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                max_num_samples=87,
                num_neg_samples=(
                    87,
                    87,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    class_text_path=
    '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/class_texts_oraldetect.json',
    dataset=dict(
        ann_file=
        '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_train.json',
        data_prefix=dict(img=''),
        data_root=
        '/ix/lzhan/siyuan/datasets/processed_datas/OralDetect_Family/OralDetect_data_by_modality/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'abnormal oral epithelial cell',
                'alveolar bone loss',
                'anterior teeth with fenestration or dehiscence',
                'anterior teeth without fenestration or dehiscence',
                'canine',
                'central incisor',
                'craniofacial or oral anomaly',
                'dental abrasion',
                'dental caries',
                'dental crown restoration',
                'dental filling',
                'dental implant',
                'dental opacity',
                'dental plaque',
                'dental restoration',
                'dental restoration (filling or crown)',
                'dividing oral cell',
                'first molar',
                'first premolar',
                'foreign object or debris',
                'hard recognized oral cell',
                'impacted tooth',
                'intraoral appliance',
                'lateral incisor',
                'lightly abnormal oral cell',
                'malignant oral cell',
                'mandibular canal',
                'maxillary sinus',
                'missing or residual root',
                'missing teeth',
                'normal',
                'normal oral cell',
                'oral blood cell',
                'orthodontic bracket',
                'periapical lesion',
                'periodontal pocket',
                'primary endodontic lesion',
                'primary endodontic with secondary periodontal lesion',
                'primary periodontal lesion',
                'primary periodontal with secondary endodontic lesion',
                'quadrant 1 (upper right)',
                'quadrant 2 (upper left)',
                'quadrant 3 (lower left)',
                'quadrant 4 (lower right)',
                'reactive oral cell',
                'retained root',
                'root canal treatment',
                'second molar',
                'second premolar',
                'severe gingivitis',
                'suspicious malignant oral cell',
                'tooth 11 (upper right central incisor)',
                'tooth 12 (upper right lateral incisor)',
                'tooth 13 (upper right canine)',
                'tooth 14 (upper right first premolar)',
                'tooth 15 (upper right second premolar)',
                'tooth 16 (upper right first molar)',
                'tooth 17 (upper right second molar)',
                'tooth 18 (upper right third molar)',
                'tooth 21 (upper left central incisor)',
                'tooth 22 (upper left lateral incisor)',
                'tooth 23 (upper left canine)',
                'tooth 24 (upper left first premolar)',
                'tooth 25 (upper left second premolar)',
                'tooth 26 (upper left first molar)',
                'tooth 27 (upper left second molar)',
                'tooth 28 (upper left third molar)',
                'tooth 31 (lower left central incisor)',
                'tooth 32 (lower left lateral incisor)',
                'tooth 33 (lower left canine)',
                'tooth 34 (lower left first premolar)',
                'tooth 35 (lower left second premolar)',
                'tooth 36 (lower left first molar)',
                'tooth 37 (lower left second molar)',
                'tooth 38 (lower left third molar)',
                'tooth 41 (lower right central incisor)',
                'tooth 42 (lower right lateral incisor)',
                'tooth 43 (lower right canine)',
                'tooth 44 (lower right first premolar)',
                'tooth 45 (lower right second premolar)',
                'tooth 46 (lower right first molar)',
                'tooth 47 (lower right second molar)',
                'tooth 48 (lower right third molar)',
                'tooth erosion',
                'tooth malformation',
                'true combined endo-perio lesion',
                'tumor',
            )),
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                1024,
                1024,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='MultiModalMosaic'),
        dict(
            border=(
                -512,
                -512,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100.0,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.5,
                1.5,
            ),
            type='WeDetectRandomAffine'),
        dict(
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    img_scale=(
                        1024,
                        1024,
                    ),
                    pad_val=114.0,
                    pre_transform=[
                        dict(backend_args=None, type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ],
                    type='MultiModalMosaic'),
                dict(
                    border=(
                        -512,
                        -512,
                    ),
                    border_val=(
                        114,
                        114,
                        114,
                    ),
                    max_aspect_ratio=100.0,
                    max_rotate_degree=0.0,
                    max_shear_degree=0.0,
                    scaling_ratio_range=(
                        0.5,
                        1.5,
                    ),
                    type='WeDetectRandomAffine'),
            ],
            prob=0.15,
            type='YOLOv5MultiModalMixUp'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='WeDetectHSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=87,
            num_neg_samples=(
                87,
                87,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            1024,
            1024,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(
        border=(
            -512,
            -512,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100.0,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='WeDetectRandomAffine'),
    dict(
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    1024,
                    1024,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='MultiModalMosaic'),
            dict(
                border=(
                    -512,
                    -512,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100.0,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='WeDetectRandomAffine'),
        ],
        prob=0.15,
        type='YOLOv5MultiModalMixUp'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='WeDetectHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=87,
        num_neg_samples=(
            87,
            87,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        1024,
        1024,
    ), type='WeDetectKeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            1024,
            1024,
        ),
        type='WeDetectLetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='WeDetectRandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='WeDetectHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=87,
        num_neg_samples=(
            87,
            87,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path=
        '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/class_texts_oraldetect.json',
        dataset=dict(
            ann_file=
            '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img=''),
            data_root=
            '/ix/lzhan/siyuan/datasets/processed_datas/OralDetect_Family/OralDetect_data_by_modality/',
            metainfo=dict(
                classes=(
                    'abnormal oral epithelial cell',
                    'alveolar bone loss',
                    'anterior teeth with fenestration or dehiscence',
                    'anterior teeth without fenestration or dehiscence',
                    'canine',
                    'central incisor',
                    'craniofacial or oral anomaly',
                    'dental abrasion',
                    'dental caries',
                    'dental crown restoration',
                    'dental filling',
                    'dental implant',
                    'dental opacity',
                    'dental plaque',
                    'dental restoration',
                    'dental restoration (filling or crown)',
                    'dividing oral cell',
                    'first molar',
                    'first premolar',
                    'foreign object or debris',
                    'hard recognized oral cell',
                    'impacted tooth',
                    'intraoral appliance',
                    'lateral incisor',
                    'lightly abnormal oral cell',
                    'malignant oral cell',
                    'mandibular canal',
                    'maxillary sinus',
                    'missing or residual root',
                    'missing teeth',
                    'normal',
                    'normal oral cell',
                    'oral blood cell',
                    'orthodontic bracket',
                    'periapical lesion',
                    'periodontal pocket',
                    'primary endodontic lesion',
                    'primary endodontic with secondary periodontal lesion',
                    'primary periodontal lesion',
                    'primary periodontal with secondary endodontic lesion',
                    'quadrant 1 (upper right)',
                    'quadrant 2 (upper left)',
                    'quadrant 3 (lower left)',
                    'quadrant 4 (lower right)',
                    'reactive oral cell',
                    'retained root',
                    'root canal treatment',
                    'second molar',
                    'second premolar',
                    'severe gingivitis',
                    'suspicious malignant oral cell',
                    'tooth 11 (upper right central incisor)',
                    'tooth 12 (upper right lateral incisor)',
                    'tooth 13 (upper right canine)',
                    'tooth 14 (upper right first premolar)',
                    'tooth 15 (upper right second premolar)',
                    'tooth 16 (upper right first molar)',
                    'tooth 17 (upper right second molar)',
                    'tooth 18 (upper right third molar)',
                    'tooth 21 (upper left central incisor)',
                    'tooth 22 (upper left lateral incisor)',
                    'tooth 23 (upper left canine)',
                    'tooth 24 (upper left first premolar)',
                    'tooth 25 (upper left second premolar)',
                    'tooth 26 (upper left first molar)',
                    'tooth 27 (upper left second molar)',
                    'tooth 28 (upper left third molar)',
                    'tooth 31 (lower left central incisor)',
                    'tooth 32 (lower left lateral incisor)',
                    'tooth 33 (lower left canine)',
                    'tooth 34 (lower left first premolar)',
                    'tooth 35 (lower left second premolar)',
                    'tooth 36 (lower left first molar)',
                    'tooth 37 (lower left second molar)',
                    'tooth 38 (lower left third molar)',
                    'tooth 41 (lower right central incisor)',
                    'tooth 42 (lower right lateral incisor)',
                    'tooth 43 (lower right canine)',
                    'tooth 44 (lower right first premolar)',
                    'tooth 45 (lower right second premolar)',
                    'tooth 46 (lower right first molar)',
                    'tooth 47 (lower right second molar)',
                    'tooth 48 (lower right third molar)',
                    'tooth erosion',
                    'tooth malformation',
                    'true combined endo-perio lesion',
                    'tumor',
                )),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1024,
                1024,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1024,
                    1024,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataset = dict(
    class_text_path=
    '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/class_texts_oraldetect.json',
    dataset=dict(
        ann_file=
        '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root=
        '/ix/lzhan/siyuan/datasets/processed_datas/OralDetect_Family/OralDetect_data_by_modality/',
        metainfo=dict(
            classes=(
                'abnormal oral epithelial cell',
                'alveolar bone loss',
                'anterior teeth with fenestration or dehiscence',
                'anterior teeth without fenestration or dehiscence',
                'canine',
                'central incisor',
                'craniofacial or oral anomaly',
                'dental abrasion',
                'dental caries',
                'dental crown restoration',
                'dental filling',
                'dental implant',
                'dental opacity',
                'dental plaque',
                'dental restoration',
                'dental restoration (filling or crown)',
                'dividing oral cell',
                'first molar',
                'first premolar',
                'foreign object or debris',
                'hard recognized oral cell',
                'impacted tooth',
                'intraoral appliance',
                'lateral incisor',
                'lightly abnormal oral cell',
                'malignant oral cell',
                'mandibular canal',
                'maxillary sinus',
                'missing or residual root',
                'missing teeth',
                'normal',
                'normal oral cell',
                'oral blood cell',
                'orthodontic bracket',
                'periapical lesion',
                'periodontal pocket',
                'primary endodontic lesion',
                'primary endodontic with secondary periodontal lesion',
                'primary periodontal lesion',
                'primary periodontal with secondary endodontic lesion',
                'quadrant 1 (upper right)',
                'quadrant 2 (upper left)',
                'quadrant 3 (lower left)',
                'quadrant 4 (lower right)',
                'reactive oral cell',
                'retained root',
                'root canal treatment',
                'second molar',
                'second premolar',
                'severe gingivitis',
                'suspicious malignant oral cell',
                'tooth 11 (upper right central incisor)',
                'tooth 12 (upper right lateral incisor)',
                'tooth 13 (upper right canine)',
                'tooth 14 (upper right first premolar)',
                'tooth 15 (upper right second premolar)',
                'tooth 16 (upper right first molar)',
                'tooth 17 (upper right second molar)',
                'tooth 18 (upper right third molar)',
                'tooth 21 (upper left central incisor)',
                'tooth 22 (upper left lateral incisor)',
                'tooth 23 (upper left canine)',
                'tooth 24 (upper left first premolar)',
                'tooth 25 (upper left second premolar)',
                'tooth 26 (upper left first molar)',
                'tooth 27 (upper left second molar)',
                'tooth 28 (upper left third molar)',
                'tooth 31 (lower left central incisor)',
                'tooth 32 (lower left lateral incisor)',
                'tooth 33 (lower left canine)',
                'tooth 34 (lower left first premolar)',
                'tooth 35 (lower left second premolar)',
                'tooth 36 (lower left first molar)',
                'tooth 37 (lower left second molar)',
                'tooth 38 (lower left third molar)',
                'tooth 41 (lower right central incisor)',
                'tooth 42 (lower right lateral incisor)',
                'tooth 43 (lower right canine)',
                'tooth 44 (lower right first premolar)',
                'tooth 45 (lower right second premolar)',
                'tooth 46 (lower right first molar)',
                'tooth 47 (lower right second molar)',
                'tooth 48 (lower right third molar)',
                'tooth erosion',
                'tooth malformation',
                'true combined endo-perio lesion',
                'tumor',
            )),
        test_mode=True,
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1024,
            1024,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1024,
                1024,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
val_evaluator = dict(
    ann_file=
    '/ihome/lzhan/sid51/projects/VLM/OralDetect-Family/OralDetect/datas/instances_oraldetect_val.json',
    metric='bbox',
    type='PerModalityCocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.05
