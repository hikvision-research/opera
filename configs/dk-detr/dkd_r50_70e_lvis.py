_base_ = [
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='opera.DKDETR',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.DKDETRHead',
        num_query=300,
        num_classes=1203,
        in_channels=2048,
        sync_cls_avg_factor=True,
        dynamic_params_dims=441,
        dynamic_encoder_heads=4,
        mask_positional_encoding_cfg=dict(
            type='opera.RelSinePositionalEncoding',
            num_feats=4,
            normalize=True),
        dice_mask_loss_weight=8.0,
        bce_mask_loss_weight=2.0,
        transformer=dict(
            type='opera.SOITTransformer',
            mask_channels=8,
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            seg_encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=1,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='mmcv.DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0)),
    text_encoder=dict(
        type='PseudoTextEncoder',
        text_dim=512,
        text_feat_path='checkpoints/dk-dter/lvis_text_embedding.pt'),
    temperature=0.05,
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='mmdet.HungarianAssigner',
            cls_cost=dict(
                type='mmdet.FocalLossCost', weight=2.0),
            reg_cost=dict(
                type='mmdet.BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(
                type='mmdet.IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=300))

image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize',
         img_scale=image_size,
         ratio_range=(0.1, 2),
         keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=image_size),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size=image_size),
    dict(type='mmdet.DefaultFormatBundle'),
    dict(type='mmdet.Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size=image_size),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=None,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])

# optimizer
optimizer = dict(
    type='AdamW',
    lr=8e-4,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)},
        norm_decay_mult=0.,
        bypass_duplicate=True
    ))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[55])
runner = dict(type='EpochBasedRunner', max_epochs=70)
checkpoint_config = dict(interval=1, create_symlink=False, max_keep_ckpts=20)

dist_params = dict(backend='nccl')

load_from = None
