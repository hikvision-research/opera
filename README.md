## Introduction

**O**bject **Per**ception & **A**pplication (Opera) is a unified toolbox for multiple computer vision tasks: detection, segmentation, pose estimation, etc.

To date, Opera implements the following algorithms:

- [PETR (CVPR'2022)](configs/petr)
- [SOIT (AAAI'2022)](configs/soit)
- [InsPose (ACM MM'2021)](configs/inspose)

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Environment

- Linux
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.1+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [MMDetection](https://mmdet.readthedocs.io/en/latest/#installation)

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of Opera.

## Acknowledgement

Opera is an open source project built upon [OpenMMLab](https://github.com/open-mmlab/). We appreciate all the contributors who implement this flexible and efficient toolkits.

## Citations

If you find our works useful in your research, please consider citing:
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

@inproceedings{shi2021inspose,
  title={Inspose: instance-aware networks for single-stage multi-person pose estimation},
  author={Shi, Dahu and Wei, Xing and Yu, Xiaodong and Tan, Wenming and Ren, Ye and Pu, Shiliang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3079--3087},
  year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
