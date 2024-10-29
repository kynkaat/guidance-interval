"""EDM sampler with optional guidance interval."""
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
import torch
from typing import Any, \
                   Optional, \
                   Tuple


def edm_sampler(cond_net: Any,
                latents: torch.Tensor,
                uncond_net: Optional[Any] = None,
                class_labels: Optional[torch.Tensor] = None,
                G: float = 1.0,
                guidance_interval: Optional[list] = None,
                num_steps: int = 32,
                sigma_min: float = 0.002,
                sigma_max: float = 80.0,
                rho: float = 7.0,
                S_churn: float = 0.0,
                S_min: float = 0.0,
                S_max: float = float('inf'),
                S_noise: float = 1.0,
                verbose: bool = False) -> torch.Tensor:
    """Second order EDM sampler."""
    # Determine denoising function.
    if guidance_interval is None:  # Regular sampling mode. If uncond. net is available, then guidance is used.
        if uncond_net is None:  # No guidance.
            denoise_fn = lambda *args: cond_net(*args).to(torch.float64)
        else:  # Apply guidance.
            denoise_fn = lambda *args: uncond_net(*args).to(torch.float64).lerp(cond_net(*args).to(torch.float64), G)
    else:
        guidance_start, guidance_stop = guidance_interval

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, cond_net.sigma_min)
    sigma_max = min(sigma_max, cond_net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([cond_net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = cond_net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        if verbose:
            print(f'sigma={t_hat:0.3f}')

        # Update denoising function on the fly when using guidance interval.
        if guidance_interval is not None:
            if i >= guidance_start and i <= guidance_stop:
                if verbose:
                    print(f'Using guidance (G = {G})...')
                denoise_fn = lambda *args: uncond_net(*args).to(torch.float64).lerp(cond_net(*args).to(torch.float64), G)
            else:
                if verbose:
                    print('No guidance...')
                denoise_fn = lambda *args: cond_net(*args).to(torch.float64)

        # Euler step.
        denoised = denoise_fn(x_hat, t_hat, class_labels)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = denoise_fn(x_next, t_next, class_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next.to(torch.float32)
