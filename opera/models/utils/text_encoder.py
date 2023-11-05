import os

import numpy as np
import torch
from mmcv.runner import BaseModule
from mmdet.datasets import CocoDataset, LVISDataset, VOCDataset
from transformers import CLIPProcessor, CLIPModel

from .builder import TEXT_ENCODER
from opera.datasets import Objects365


vild_templates = (
    'There is a {} in the scene.',
    'There is the {} in the scene.',
    'a photo of a {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',
    'itap of a {}.',
    'itap of my {}.',
    'itap of the {}.',
    'a photo of a {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',
    'a good photo of a {}.',
    'a good photo of the {}.',
    'a bad photo of a {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',
    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',
    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',
    'a bright photo of a {}.',
    'a bright photo of the {}.',
    'a dark photo of a {}.',
    'a dark photo of the {}.',
    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of a {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of a {}.',
    'a cropped photo of the {}.',
    'a close-up photo of a {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of a {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of a {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of a {}.',
    'a pixelated photo of the {}.',
    'a black and white photo of the {}.',
    'a black and white photo of a {}.',
    'a plastic {}.',
    'the plastic {}.',
    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',
    'an embroidered {}.',
    'the embroidered {}.',
    'a painting of the {}.',
    'a painting of a {}.'
    )


class BaseTextEncoder(BaseModule):

    def __init__(self, text_dim=512, init_cfg=None):
        super().__init__(init_cfg)
        self.text_dim = text_dim

    def get_text_feat(self, device='cpu'):
        raise NotImplementedError


@TEXT_ENCODER.register_module()
class PseudoTextEncoder(BaseTextEncoder):
    """
    A pseudo text encoder that produce a group of fixed text features.

    Args:
        text_feat_path (str): the path of fixed text features.
    """

    def __init__(self, text_feat_path='', **kwargs):
        super().__init__(**kwargs)
        assert isinstance(text_feat_path, str)
        assert os.path.isfile(text_feat_path)
        # Load off-the-shelf text features
        if text_feat_path.endswith('.npy'):
            text_feat = torch.tensor(
                np.load(text_feat_path), dtype=torch.float32)
        elif text_feat_path.endswith('.pt'):
            text_feat = torch.load(text_feat_path).to(torch.float32)
        else:
            raise TypeError(
                f'Unsupported file type: {text_feat_path.split(".")[-1]}')
        self.register_buffer('text_feat', text_feat)

    def get_text_feat(self, device='cpu'):
        text_feat = getattr(self, 'text_feat').to(device)
        return text_feat


@TEXT_ENCODER.register_module()
class CLIPTextEncoder(BaseTextEncoder):

    def __init__(self,
                 pretrained,
                 dataset='lvis',
                 prompts=vild_templates,
                 **kwargs):
        assert dataset in ['lvis', 'coco', 'obj365', 'voc']
        if dataset == 'lvis':
            class_names = LVISDataset.CLASSES
        elif dataset == 'coco':
            class_names = CocoDataset.CLASSES
        elif dataset == 'obj365':
            class_names = Objects365.CLASSES
        elif dataset == 'voc':
            class_names = VOCDataset.CLASSES
        else:
            raise KeyError(f'Unexpected dataset: {dataset}')
        super().__init__(**kwargs)
        model = CLIPModel.from_pretrained(pretrained)
        self.processor = CLIPProcessor.from_pretrained(pretrained)
        self.text_dim = model.text_embed_dim
        self.model = model.text_model
        self.projection = model.text_projection
        self.freeze_model()

        # assemble names with prompts
        assert isinstance(prompts, (list, tuple))
        self.num_prompt = len(prompts)
        text = [[prompt.format(name) for name in class_names]
                for prompt in prompts]
        self.text = [x for xx in text for x in xx]

    def init_weights(self):
        pass

    def freeze_model(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, device='cpu'):
        batch_inputs = self.processor(
            text=text,
            return_tensors='pt',
            truncation=True,
            padding=True).to(device)
        pooler_output = self.model(**batch_inputs).pooler_output
        text_feat = self.projection(pooler_output)
        return text_feat

    def get_text_feat(self, device='cpu'):
        # generating only once
        if getattr(self, 'text_feat', None) is None:
            text_feat = self(self.text, device=device)
            text_feat = text_feat.reshape(
                self.num_prompt, -1, self.text_dim).mean(0)
            setattr(self, 'text_feat', text_feat.cpu())
        else:
            text_feat = getattr(self, 'text_feat').to(device)
        return text_feat
