# End-to-End Multi-Person Pose Estimation with Transformers


## Results and Models

| Model | Backbone | Lr schd | mAP  | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | Download |
|:-----:|:--------:|:-------:|:----:|:---------------:|:---------------:|:--------------:|:--------------:|:--------:|
| PETR  |  R-50    |  100e   | 68.8 |      87.5       |      76.3       |      62.7      |      77.7      |[model](https://drive.google.com/file/d/1HcwraqWdZ3CaGMQOJHY8exNem7UnFkfS/view?usp=sharing) |
| PETR  |  R-101   |  100e   | 70.0 |      88.5       |      77.5       |      63.6      |      79.4      |[model](https://drive.google.com/file/d/1O261Jrt4JRGlIKTmLtPy3AUruwX1hsDf/view?usp=sharing) |
| PETR  |  Swin-L  |  100e   | 73.1 |      90.7       |      80.9       |      67.2      |      81.7      |[model](https://drive.google.com/file/d/1ujL0Gm5tPjweT0-gdDGkTc7xXrEt6gBP/view?usp=sharing) |

# NOTE

1. Swin-L are trained with batch size 16 due to GPU memory limitation.
2. The performance is unstable. `PETR` may fluctuate about 0.2 mAP.
