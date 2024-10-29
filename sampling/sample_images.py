"""Functions for sampling images."""

import numpy as np
import torch
from typing import Any, \
                   List, \
                   Optional

from sampling.edm_sampler import edm_sampler


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
class StackedRandomGenerator:
    """Wrapper for torch.Generator that allows specifying a different random seed for each sample in a minibatch.
    Adapted from: https://github.com/NVlabs/edm/blob/main/generate.py"""
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def denoise_latents(cond_net: Any,
                    uncond_net: Any,
                    seeds: List[int],
                    G: float,
                    sampler_kwargs: dict,
                    class_label: Optional[int] = None,
                    guidance_interval: Optional[List[int]] = None,
                    batch_size: int = 64,
                    device: torch.device = torch.device('cuda'),
                    verbose: bool = False) -> torch.Tensor:
    """Denoise latents with EDM sampler."""
    assert cond_net.label_dim
    label_dim = cond_net.label_dim
    num_latents = len(seeds)

    if class_label:
        # Use hashed seeds for qualitative visualizations.
        seeds = [hash((class_label, seed)) % (1 << 31) for seed in seeds]

    # Denoise latents.
    denoised_latents = torch.zeros([num_latents,
                                    cond_net.img_channels,
                                    cond_net.img_resolution,
                                    cond_net.img_resolution],
                                   device=device,
                                   dtype=torch.float32)

    for begin in range(0, num_latents, batch_size):
        end = min(begin + batch_size, num_latents)

        # Sample latents.
        rng = StackedRandomGenerator(device=device,
                                     seeds=seeds[begin:end])
        latents = rng.randn(size=[end - begin,
                                  cond_net.img_channels,
                                  cond_net.img_resolution,
                                  cond_net.img_resolution], device=device)

        if class_label:
            # Use fixed class labels.
            labels_np = (class_label * np.ones(end - begin)).astype(np.int32)
            class_labels = torch.eye(label_dim, device=device)[labels_np]
        else:
            # Sample class labels.
            class_labels = torch.eye(label_dim, device=device)[rng.randint(label_dim, size=[end - begin], device=device)]

        denoised_latents[begin:end] = edm_sampler(cond_net=cond_net,
                                                  latents=latents,
                                                  uncond_net=uncond_net,
                                                  class_labels=class_labels,
                                                  G=G,
                                                  guidance_interval=guidance_interval,
                                                  verbose=verbose,
                                                  **sampler_kwargs)
    return denoised_latents


def decode_latents(denoised_latents: torch.Tensor,
                   vae: Any,
                   batch_size: int = 8,
                   img_channels: int = 3,
                   resolution: int = 512,
                   device: torch.device = torch.device('cuda')) -> np.ndarray:
    """Decodes denoised latents to RGB images."""
    _vae_params = dict(scale=[0.11990407, 0.10822511, 0.13477089, 0.15243903],
                       bias=[-0.69664264, -0.3517316, -0.016172506, 0.32774392])

    # Decode latents.
    num_latents = denoised_latents.shape[0]
    images = np.zeros([num_latents, img_channels, resolution, resolution], dtype=np.uint8)
    scale = torch.as_tensor(_vae_params['scale'], dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
    bias = torch.as_tensor(_vae_params['bias'], dtype=torch.float32, device=device).reshape(1, -1, 1, 1)

    for begin in range(0, num_latents, batch_size):
        end = min(begin + batch_size, num_latents)
        latent_batch = (denoised_latents[begin:end] - bias) / scale

        with torch.no_grad():
            images[begin:end] = vae.decode(latent_batch)['sample'].clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()

    return images
