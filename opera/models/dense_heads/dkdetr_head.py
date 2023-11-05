import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, Linear
from mmdet.core import bbox_cxcywh_to_xyxy

from ..builder import HEADS
from .soit_head import SOITHead, aligned_bilinear


def get_cosine_similarity(x, y):
    """
    Calculate cosine similarity between two tensors.

    Args:
        x (torch.Tensor): shape (M, *, D)
        y (torch.Tensor): shape (N, D)
    """
    assert x.shape[-1] == y.shape[-1]
    assert y.ndim == 2
    y = F.normalize(y.permute(1, 0).contiguous(), p=2, dim=0)
    x = F.normalize(x, p=2, dim=-1)

    if x.ndim > 2:
        ori_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        sim = torch.mm(x, y)
        sim = sim.reshape(*ori_shape[:-1], sim.shape[-1])
    else:
        sim = torch.mm(x, y)

    return sim


@HEADS.register_module()
class DKDETRHead(SOITHead):

    def __init__(self,
                 *args,
                 output_mask=True,
                 text_dim=512,
                 temperature=1.,
                 **kwargs):
        self.text_dim = text_dim
        self.temperature = temperature
        if 'with_box_refine' in kwargs:
            kwargs.pop('with_box_refine')
        if 'as_two_stage' in kwargs:
            kwargs.pop('as_two_stage')
        super().__init__(
            *args,
            with_box_refine=True,
            as_two_stage=True,
            **kwargs)
        assert self.as_two_stage
        assert self.with_box_refine
        self.output_mask = output_mask

    def _init_layers(self):
        self.cls_out_channels = self.text_dim
        super()._init_layers()
        num_dec = self.transformer.decoder.num_layers
        self.cls_branches = self.cls_branches[:num_dec]
        self.cls_branches.append(Linear(self.embed_dims, self.num_classes))

    def init_weights(self):
        self.transformer.init_weights()
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        for m in self.reg_branches:
            nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas, text_feats):
        outputs = super().forward(mlvl_feats, img_metas)
        outputs_classes = outputs[0]
        outputs_classes = get_cosine_similarity(
            outputs_classes, text_feats) / self.temperature
        return (outputs_classes, ) + outputs[1:]

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           dynamic_params,
                           seg_memory,
                           seg_pos_embed,
                           seg_mask,
                           seg_reference_points,
                           level_start_index,
                           spatial_shapes,
                           img_shape,
                           ori_shape,
                           scale_factor,
                           rescale=False):
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        mask_thr = self.test_cfg.get('mask_thresh', 0.5)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexs = cls_score.view(-1).topk(max_per_img)
            det_labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        if self.output_mask:
            dynamic_params = dynamic_params[bbox_index]
            cxcy = bbox_pred[:, :2]
            img_mask = self.p3_mask
            seg_pos_embeds = []
            num_res = dynamic_params.size(0)
            for i in range(dynamic_params.size(0)):
                seg_pos_embed = self.mask_positional_encoding(
                    img_mask, cxcy[i])
                seg_pos_embed = seg_pos_embed.flatten(2).transpose(
                    1, 2).permute(1, 0, 2)
                seg_pos_embeds.append(seg_pos_embed)
            seg_pos_embeds = torch.cat(seg_pos_embeds, dim=1)
            seg_memory = seg_memory.repeat(1, num_res, 1)
            seg_reference_points = seg_reference_points.repeat(
                num_res, 1, 1, 1)
            mask_preds = self.dynamic_encoder.forward_test(
                dynamic_params,
                seg_memory,
                None,
                None,
                query_pos=seg_pos_embeds,
                key_padding_mask=seg_mask,
                reference_points=seg_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
            h, w = spatial_shapes[0]
            mask_preds = mask_preds.squeeze().reshape(num_res, h, w)

            pad_mask = seg_mask.reshape(1, 1, h, w).float()
            pad_mask = F.interpolate(
                pad_mask,
                scale_factor=4,
                mode='bilinear',
                align_corners=True).to(torch.bool).squeeze()
            mask_preds = aligned_bilinear(mask_preds[None], factor=4).sigmoid()
            mask_preds.masked_fill(pad_mask, 0)
            det_masks = F.interpolate(
                mask_preds,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=True)
            det_masks = det_masks[..., :img_shape[0], :img_shape[1]]
            if rescale:
                det_masks = F.interpolate(
                    det_masks,
                    size=ori_shape[:2],
                    mode='bilinear',
                    align_corners=True)[0]
            else:
                det_masks = det_masks[0]
            det_masks = det_masks > mask_thr
            # BG is not included in num_classes
            cls_segms = [[] for _ in range(self.num_classes)]
            N = det_masks.size(0)
            for i in range(N):
                cls_segms[det_labels[i]].append(
                    det_masks[i].detach().cpu().numpy())
        else:
            cls_segms = []

        return det_bboxes, det_labels, cls_segms
