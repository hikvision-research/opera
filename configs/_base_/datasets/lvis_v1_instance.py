# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'mmdet.LVISV1Dataset'
data_root = '/data/dataset/object_detection/coco/images/'
ann_root = '/data/dataset/object_detection/lvis/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmdet.DefaultFormatBundle'),
    dict(type='mmdet.Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type='mmdet.ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_root + 'lvis_v1_train.json',
            img_prefix=data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'lvis_v1_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric=['bbox'])
