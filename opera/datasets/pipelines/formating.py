# Copyright (c) Hikvision Research Institute. All rights reserved.
from collections.abc import Sequence

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formating import to_tensor as to_tensor
from mmdet.datasets.pipelines.formating import DefaultFormatBundle \
    as MMDetDefaultFormatBundle

from ..builder import PIPELINES


@PIPELINES.register_module()
class DefaultFormatBundle(MMDetDefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    - gt_keypoints: (1)to tensor, (2)to DataContainer
    - gt_areas: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        results = super(DefaultFormatBundle, self).__call__(results)
        for key in ['gt_keypoints', 'gt_areas']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        return results
