_base_ = 'dkd_r50_70e_lvis.py'

model = dict(
    output_mask=False,
    text_encoder=dict(
        text_feat_path='checkpoints/dk-dter/voc_text_embedding.pt')
)

data_root = '/data/dataset/object_detection/VOCdevkit/'
data = dict(
    train=None,
    val=dict(
        type='mmdet.VOCDataset',
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/'
    ),
    test=dict(
        type='mmdet.VOCDataset',
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/'
    )
)

evaluation = dict(metric=['mAP'], iou_thr=[0.5, 0.75])
