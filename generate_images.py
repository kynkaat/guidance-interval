"""Script to reproduce qualitative ImageNet-512 results."""

import click
import numpy as np
import os
import torch
from typing import List

from fig_configs import fig_configs
from sampling.sample_images import denoise_latents, \
                                   decode_latents
from utils import create_grid, \
                  load_networks


@click.command()
@click.option('--cond_pkl', required=True, type=str, help='Path to conditional network pickle.')
@click.option('--uncond_pkl', required=True, type=str, help='Path to unconditional network pickle.')
@click.option('--figs', default='all', type=str, help='Comma separated string of figure names (see fig_configs.py).')
@click.option('--results_dir', default='./results', type=str, help='Path to results directory.')
@click.option('--verbose', default=False, type=bool, help='Flag for extra prints.')
@click.option('--seed', default=0, type=int, help='Random seed for generating images to zip.')
def main(cond_pkl: str,
         uncond_pkl: str,
         figs: str,
         results_dir: str,
         verbose: bool,
         seed: int) -> None:
    """Reproduces qualitative ImageNet-512 results.
    
    Examples:

    \b  
    Generate all figures:
    python generate_images.py --cond_pkl=<path-to-cond-pickle> \\
        --uncond_pkl=<path-to-uncond-pickle> \\
        --figs=all

    \b
    Generate figures 5 and 6:
    python generate_images.py --cond_pkl=<path-to-cond-pickle> \\
        --uncond_pkl=<path-to-uncond-pickle> \\
        --figs=fig5,fig6

    """
    device = torch.device('cuda')
    sampler_kwargs = dict(num_steps=32,
                          sigma_min=0.002,
                          sigma_max=80.0,
                          rho=7,
                          S_churn=0.0,
                          S_min=0.0,
                          S_max=float('inf'),
                          S_noise=1.0)

    # Load networks.
    cond_net, uncond_net, vae = load_networks(cond_pkl=cond_pkl,
                                              uncond_pkl=uncond_pkl,
                                              device=device)

    # Generate specified figures.
    figs = list(fig_configs.keys()) if figs == 'all' else figs.split(',')

    for fig in figs:
        print(f'Generating figure "{fig}"...')
        assert fig in fig_configs, f'Invalid figure name: {fig}.'
        config = fig_configs[fig]
        mode = config['mode']
        pidxs = config['pidxs']
        all_seeds = config['seeds']
        cur_results_dir = os.path.join(results_dir, fig)
        os.makedirs(cur_results_dir, exist_ok=True)

        fnames: List[str] = []
        all_images: List[np.ndarray] = []
        for pidx, seeds in zip(pidxs, all_seeds):
            for G, guidance_interval in config['G_config']:
                method = 'gi' if guidance_interval is not None else 'cfg'
                # Denoise and decode latents.
                with torch.no_grad():
                    denoised_latents = denoise_latents(cond_net=cond_net,
                                                       uncond_net=uncond_net,
                                                       class_label=pidx,
                                                       seeds=seeds,
                                                       G=G,
                                                       sampler_kwargs=sampler_kwargs,
                                                       guidance_interval=guidance_interval,
                                                       device=device,
                                                       verbose=verbose)
                    all_images.append(decode_latents(denoised_latents=denoised_latents, vae=vae, device=device))

                # Define grid name for constant G grid.
                if mode == 'G_const':
                    seeds_str = '_'.join([f'{seed:03d}' for seed in seeds])
                    fnames.append(f'p{pidx:03d}-s{seeds_str}-g{G:0.1f}-{method}.png')

            # Define grid name for G sweep grid.
            if mode == 'G_sweep':
                fnames.append(f'p{pidx:03d}-s{seeds[0]:03d}-{method}.png')

        # Reshape for G sweep grid.
        if mode == 'G_sweep':
            all_images = np.concatenate(all_images, axis=0)
            all_images = all_images.reshape(len(pidxs), all_images.shape[0] // len(pidxs), 3, 512, 512)

        # Save grids.
        insets = config.get('insets', None)
        for fname, images in zip(fnames, all_images):
            create_grid(grid_path=os.path.join(cur_results_dir, fname),
                        images=images,
                        inset=insets.pop(0) if insets is not None else None,
                        **config['grid_config'])

    # All good.
    print('Done.')


if __name__ == "__main__":
    main()
