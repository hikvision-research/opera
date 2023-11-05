_base_ = 'dkd_r50_70e_lvis.py'

model = dict(
    output_mask=False,
    text_encoder=dict(
        text_feat_path='checkpoints/dk-dter/obj365_text_embedding.pt')
)

data_root = 'dataset/objects365_2M/'
ann_root = 'dataset/obj365/'
data = dict(
    train=None,
    val=dict(
        type='opera.Objects365',
        ann_file=ann_root+'zhiyuan_objv2_val.json',
        img_prefix=data_root + 'images/val'),
    test=dict(
        type='opera.Objects365',
        ann_file=ann_root+'zhiyuan_objv2_val.json',
        img_prefix=data_root + 'images/val'))

evaluation = dict(metric=['bbox'])
