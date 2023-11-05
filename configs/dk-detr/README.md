# [Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.pdf)

## Results and Models

### LVIS

We train the model on LVIS dataset with only base-category annotations, and validate the model on `LVIS v1 val` with both base and novel categories. The [text prompts](https://drive.google.com/file/d/1PMPvEWYLi2Kp2wgIiMR8m9r4mLDkxyUJ/view?usp=sharing), provided by [DetPro](https://github.com/dyabel/detpro), used for LVIS dataset is same as in ViLD.

| Model | mask AP<sub>r</sub> / AP<sub>c</sub> / AP<sub>f</sub> / AP | bbox AP<sub>r</sub> / AP<sub>c</sub> / AP<sub>f</sub> / AP | Config | Text Prompt | Download |
|:-----:|:--------:|:-------:|:----:|:------:|:---------------:|
| DK-DETR | 20.5 / 29.0 / 35.3 / 30.0 | 22.4 / 31.9 / 40.1 / 33.5 | [config](https://github.com/hikvision-research/opera/blob/main/configs/dk-detr/dkd_r50_70e_lvis.py) | [Google Drive](https://drive.google.com/file/d/1PMPvEWYLi2Kp2wgIiMR8m9r4mLDkxyUJ/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1D5QySycCwE2tag-ApkkvmzNI-uK6FZmd/view) \| [BaiduYun](https://pan.baidu.com/s/1_xJLJJ_umsVzH_egSGnP9g?pwd=sibf) |

### Generalization Ability

To demonstrate the generalization ability of the open-vocabulary object detection model, we directly evaluate the LVIS-trained model on COCO, Objects365 and Pascal VOC datasets.

|  Model  |  Dataset   | AP  | AP<sup>50</sup> | AP<sup>75</sup> | Config | Text Prompt | Download |
| :-----: | :--------: | :-----: | :-------: | :--: | :-------------: | :----: | :-------------: |
| DK-DETR |    COCO    | 39.3 |      54.5      |      42.8   | [config](https://github.com/hikvision-research/opera/blob/main/configs/dk-detr/dkd_r50_70e_test_coco.py) | [Google Drive](https://drive.google.com/file/d/1RFnrZgz-Gg4-oQArMAG_IllU60MyNCXL/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1D5QySycCwE2tag-ApkkvmzNI-uK6FZmd/view) \| [BaiduYun](https://pan.baidu.com/s/1_xJLJJ_umsVzH_egSGnP9g?pwd=sibf) |
| DK-DETR | Objects365 | 13.0 |      17.9   |      13.9   | [config](https://github.com/hikvision-research/opera/blob/main/configs/dk-detr/dkd_r50_70e_test_obj365.py) | [Google Drive](https://drive.google.com/file/d/170G58VF5AQS81KNG-_6gf5wXEOZdpD6U/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1D5QySycCwE2tag-ApkkvmzNI-uK6FZmd/view) \| [BaiduYun](https://pan.baidu.com/s/1_xJLJJ_umsVzH_egSGnP9g?pwd=sibf) |
| DK-DETR | Pascal VOC | - | 71.1 | 61.3 | [config](https://github.com/hikvision-research/opera/blob/main/configs/dk-detr/dkd_r50_70e_test_voc.py) | [Google Drive](https://drive.google.com/file/d/1jqRPwbxhL4Yi-Kttn8y8E8jkhEmyFKqP/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1D5QySycCwE2tag-ApkkvmzNI-uK6FZmd/view) \| [BaiduYun](https://pan.baidu.com/s/1_xJLJJ_umsVzH_egSGnP9g?pwd=sibf) |
## Citation

```BibTeX
@inproceedings{li2023distilling,
  title={Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection},
  author={Li, Liangqi and Miao, Jiaxu and Shi, Dahu and Tan, Wenming and Ren, Ye and Yang, Yi and Pu, Shiliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6501--6510},
  year={2023}
}
```
