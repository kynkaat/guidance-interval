"""Inception-V3 and DINOv2 feature networks."""

import pickle
import PIL.Image
import torch
import torchvision
from typing import Any

import dnnlib


class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)
        self.feature_dim = 2048

    def forward(self,
                x: torch.Tensor,
                **kwargs) -> torch.Tensor:
        return self.model(x, return_features=True)


class DINOv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14')
        self.resize = torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.feature_dim = 1024

    def forward(self,
                x: torch.Tensor,
                **kwargs) -> torch.Tensor:
        # Pre-process images.
        # NOTE: Slow since tensors are first converted back to PIL.Images.
        # This is done because resizing PIL.Images and torch.tensors with BICUBIC
        # leads to slightly different results.
        x_np = [x_i.detach().cpu().numpy() for x_i in x]
        x_pils = [PIL.Image.fromarray(img.transpose(1, 2, 0)) for img in x_np]
        x_proc = [self.normalize(self.to_tensor(self.resize(x_pil))).unsqueeze(0) for x_pil in x_pils]
        x_proc = torch.cat(x_proc, dim=0).to(self.device)

        return self.model(x_proc)


def load_feature_network(name: str) -> Any:
    """Loads a feature network."""
    assert name in ['inception_v3', 'dinov2'], f'Invalid feature network name: {name}.'
    if name == 'inception_v3':
        return InceptionV3()
    else:
        return DINOv2()
