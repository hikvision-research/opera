# End-to-End Multi-Person Pose Estimation with Transformers


## Results and Models

| Model | Backbone | Lr schd  | keypoint AP | Download |
|:-----:|:--------:|:--------:|:-----------:|:--------:|
| PETR  |  R-50    |  100e    |    68.9     | [model](https://drive.google.com/file/d/1HcwraqWdZ3CaGMQOJHY8exNem7UnFkfS/view?usp=sharing) |
| PETR  |  R-101   |  100e    |    70.0     | [model](https://drive.google.com/file/d/1O261Jrt4JRGlIKTmLtPy3AUruwX1hsDf/view?usp=sharing) |
| PETR  |  Swin-L  |  100e    |    73.1     | [model](https://drive.google.com/file/d/1ujL0Gm5tPjweT0-gdDGkTc7xXrEt6gBP/view?usp=sharing) |

# NOTE

1. All models are trained with batch size 32, except swin-l backbone.
2. The performance is unstable. `PETR` may fluctuate about 0.3 mAP.
