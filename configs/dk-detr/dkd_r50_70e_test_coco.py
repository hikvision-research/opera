_base_ = 'dkd_r50_70e_lvis.py'

model = dict(
    output_mask=False,
    text_encoder=dict(
        text_feat_path='checkpoints/dk-dter/coco_text_embedding.pt')
)

data_root = '/data/dataset/object_detection/coco/'
data = dict(
    train=None,
    val=dict(
        type='mmdet.CocoDataset',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/'),
    test=dict(
        type='mmdet.CocoDataset',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/'))

evaluation = dict(metric=['bbox'])
