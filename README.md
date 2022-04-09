## Introduction

**O**bject **Per**ception & **A**pplication (Opera) is a unified toolbox for multiple computer vision tasks: detection, segmentation, pose estimation, etc.

To date, Opera implements the following algorithms:

* [PETR](configs/petr) _to be released_
* [SOIT](configs/soit) _to be released_

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Environment

- Linux or macOS
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.1+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [MMDET](https://mmdet.readthedocs.io/en/latest/#installation)

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of Opera.

## Acknowledgement

Opera is an open source project that is built upon [OpenMMLab](https://github.com/open-mmlab/). We appreciate all the contributors who implement this flexible and efficient toolkit.

## Citation

If you find our work useful in your research, please consider citing:
```BibTeX
@inproceedings{shi2022end,
  title={End-to-End Multi-Person Pose Estimation with Transformers},
  author={Shi, Dahu and Wei, Xing and Li, Liangqi and Ren, Ye and Tan, Wenming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

@inproceedings{yu2022soit,
  title={SOIT: Segmenting Objects with Instance-Aware Transformers},
  author={Yu, Xiaodong and Shi, Dahu and Wei, Xing and Ren, Ye and Ye, Tingqun and Tan, Wenming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
