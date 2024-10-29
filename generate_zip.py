"""Script to generate image zip for reproducing quantitative ImageNet-512 results in Table 1."""

import ast
import click
from io import BytesIO
import numpy as np
import os
import PIL.Image
import torch
import time
from typing import Optional
import zipfile

from sampling.sample_images import denoise_latents, \
                                   decode_latents
from utils import load_networks


@click.command()
@click.option('--cond_pkl', required=True, type=str, help='Path to conditional network pickle.')
@click.option('--uncond_pkl', required=True, type=str, help='Path to unconditional network pickle.')
@click.option('--zip_dir', default='./results', type=str, help='Path to the directory where generated images are saved.')
@click.option('--num_images', default=50000, type=int, help='Number of images generated to the zip.')
@click.option('--batch_size', default=64, type=int, help='Batch size for the denoiser network.')
@click.option('--batch_size_decoder', default=8, type=int, help='Batch size for the decoder network.')
@click.option('--guidance_scale', default=2.0, type=float, help='Guidance scale. For FID optimal results, use G = 2.0 for guidance interval, G = 1.2 for CFG.')
@click.option('--guidance_interval', default='[17, 22]', type=str, help='List specifying guidance interval (start, stop) indices. None = CFG, FID: EDM2-XXL = [17, 22], FD_DINOv2: EDM2-XXL = [13, 19].')
@click.option('--verbose', default=False, type=bool, help='Flag for extra prints.')
@click.option('--seed', default=0, type=int, help='Random seed for generating images to zip.')
def main(cond_pkl: str,
         uncond_pkl: str,
         zip_dir: str,
         num_images: int,
         batch_size: int,
         batch_size_decoder: int,
         guidance_scale: float,
         guidance_interval: Optional[list],
         verbose: bool,
         seed: int) -> None:
    """Reproduces quantitative ImageNet-512 results.

    Example:

    \b
    EDM2-XXL FID results with guidance interval:
    python generate_zip.py --cond_pkl=<path-to-cond-pickle> \\
        --uncond_pkl=<path-to-uncond-pickle> \\
        --zip_dir=<dir-where-zip-is-saved> \\
        --guidance_interval="[17, 22]" \\
        --guidance_scale=2.0

    \b
    EDM2-XXL FID results with CFG:
    python generate_zip.py --cond_pkl=<path-to-cond-pickle> \\
        --uncond_pkl=<path-to-uncond-pickle> \\
        --zip_dir=<dir-where-zip-is-saved> \\
        --guidance_scale=1.2 \\
        --guidance_interval=None

    \b
    EDM2-XXL FD_DINOv2 results with guidance interval:
    python generate_zip.py --cond_pkl=<path-to-cond-pickle> \\
        --uncond_pkl=<path-to-uncond-pickle> \\
        --zip_dir=<dir-where-zip-is-saved> \\
        --guidance_interval="[13, 19]" \\
        --guidance_scale=2.9

    \b
    EDM2-XXL FD_DINOv2 results with CFG:
    python generate_zip.py --cond_pkl=<path-to-cond-pickle> \\
        --uncond_pkl=<path-to-uncond-pickle> \\
        --zip_dir=<dir-where-zip-is-saved> \\
        --guidance_scale=1.7 \\
        --guidance_interval=None

    """
    guidance_interval = ast.literal_eval(guidance_interval)
    method = 'gi' if guidance_interval is not None else 'cfg'
    zip_name = f"{os.path.basename(cond_pkl).split('.pkl')[0]}-s{seed}-nimg{num_images}-g{guidance_scale:0.2f}-{method}.zip"
    zip_path = os.path.join(zip_dir, zip_name)
    os.makedirs(zip_dir, exist_ok=True)
    device = torch.device('cuda')
    seeds = np.arange(seed, seed + num_images, dtype=int)
    torch.manual_seed(seed)

    # Define sampler parameters.
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

    # Generate images to zip.
    print(f'Generating {num_images} images and saving them to "{zip_path}..."')
    img_idx = 0
    with zipfile.ZipFile(zip_path, 'w') as f:
        for begin in range(0, num_images, batch_size):
            end = min(begin + batch_size, num_images)
            total_start = time.time()
            with torch.no_grad():
                den_start = time.time()
                denoised_latents = denoise_latents(cond_net=cond_net,
                                                   uncond_net=uncond_net,
                                                   seeds=seeds[begin:end],
                                                   G=guidance_scale,
                                                   batch_size=batch_size,
                                                   sampler_kwargs=sampler_kwargs,
                                                   guidance_interval=guidance_interval,
                                                   device=device)
                den_time = time.time() - den_start

                dec_start = time.time()
                images = decode_latents(denoised_latents=denoised_latents,
                                        vae=vae,
                                        batch_size=batch_size_decoder,
                                        device=device)
                dec_time = time.time() - dec_start

            saving_start = time.time()
            for image in images:
                zip_fname = f'{img_idx:06d}.png'
                im = PIL.Image.fromarray(image.transpose(1, 2, 0))
                image_file = BytesIO()
                im.save(image_file, 'PNG')
                f.writestr(zip_fname, image_file.getvalue())
                img_idx += 1
            saving_time = time.time() - saving_start

            if verbose:
                elapsed_time = time.time() - total_start
                imgs_per_second = (end - begin) / elapsed_time
                print(f'{img_idx}/{num_images} images generated ({imgs_per_second:0.2f} imgs/s)')
                print(f'Batch timing: denoising = {den_time:0.1f}s, decoding = {dec_time:0.1f}s, saving = {saving_time:0.1f}s\n')

    # All good.
    print('Done.')


if __name__ == "__main__":
    main()
