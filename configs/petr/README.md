# End-to-End Multi-Person Pose Estimation with Transformers


## Results and Models

| Model | Backbone | Lr schd  | keypoint AP | Download |
|:-----:|:--------:|:--------:|:-----------:|:--------:|
| PETR  |  R-50    |  100e    |    68.9     | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth) |
| PETR  |  R-101   |  100e    |    70.0     | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth) |
| PETR  |  Swin-L  |  100e    |    73.1     | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth) |

# NOTE

1. All models are trained with batch size 32, except swin-l backbone.
2. The performance is unstable. `PETR` may fluctuate about 0.3 mAP.
