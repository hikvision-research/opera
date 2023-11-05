# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmdet.core import bbox2result
from mmdet.models.detectors.deformable_detr import DeformableDETR

from ..builder import DETECTORS
from ..utils import build_text_encoder


@DETECTORS.register_module()
class DKDETR(DeformableDETR):
    """Implementation of `Distilling DETR with Visual-Linguistic Knowledge for
    Open-Vocabulary Object Detection`."""

    def __init__(self,
                 *args,
                 output_mask=True,
                 text_encoder=None,
                 temperature=1.,
                 **kwargs):
        assert text_encoder is not None
        assert text_encoder.get('text_dim', None) is not None
        kwargs['bbox_head'].update(dict(
            output_mask=output_mask,
            text_dim=text_encoder['text_dim'],
            temperature=temperature))
        super().__init__(*args, **kwargs)
        self.output_mask = output_mask
        self.text_encoder = build_text_encoder(text_encoder)
        
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            img (list[torch.Tensor]): List of multiple images.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray, np.ndarray]]: BBox and mask results 
                of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        text_feats = self.text_encoder.get_text_feat(img[0].device)
        self.bbox_head.num_classes = text_feats.size(0)
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas, text_feats)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_masks in bbox_list
        ]

        if not self.output_mask:
            return bbox_results

        mask_results = [res[2] for res in bbox_list]
        results = [(bbox_results, mask_results)
            for bbox_results, mask_results in zip(bbox_results, mask_results)]

        return results
