"""Miscellaneous utility functions."""

import diffusers
import json
import numpy as np
import os
import pickle
import PIL.Image
import torch
from typing import Any, \
                   List, \
                   Optional, \
                   Tuple
import zipfile

import dnnlib


def count_images_in_zip(zip_path: str) -> int:
    """Returns how many PNG files are in a zip."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        png_files = [fname for fname in zf.namelist() if fname.endswith('.png')]
    return len(png_files)


def load_hf_vae(name: str = 'stabilityai/sd-vae-ft-mse',
                device: torch.device = torch.device('cuda'),
                cache_dir: Optional[str] = None,
                verbose: bool = True) -> Any:
    """Loads HuggingFace VAE for decoding denoised latents to RGB images."""
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    if cache_dir is None:
        cache_dir = dnnlib.make_cache_dir_path('diffusers')
    if verbose:
        print(f'Loading VAE: {name}...')
    try:
        # First try with local_files_only to avoid consulting tfhub metadata if the model is already in cache.
        vae = diffusers.models.AutoencoderKL.from_pretrained(name, cache_dir=cache_dir, local_files_only=True)
    except:
        # Could not load the model from cache; try without local_files_only.
        vae = diffusers.models.AutoencoderKL.from_pretrained(name, cache_dir=cache_dir)
    vae.eval().to(device).requires_grad_(False)

    return vae


def load_networks(cond_pkl: str,
                  uncond_pkl: str,
                  device: torch.device = torch.device('cuda')) -> Any:
    """Loads all networks required to generate images."""
    # Load conditional network.
    print(f'Loading cond. network from: "{cond_pkl}"...')
    with dnnlib.util.open_url(cond_pkl) as f:
        pkl_data = pickle.load(f)
    cond_net = pkl_data['ema'].to(device)

    # Load unconditional network.
    uncond_net = None
    if uncond_pkl is not None:
        print(f'Loading uncond. network from: "{uncond_pkl}"...')
        with dnnlib.util.open_url(uncond_pkl) as f:
            uncond_net = pickle.load(f)['ema'].to(device)

    # Load decoder VAE.
    vae = load_hf_vae(device=device)

    return cond_net, uncond_net, vae


def load_reference_statistics(ref_path: str,
                              metric: str = 'fid') -> Tuple[np.ndarray, np.ndarray]:
    """Loads reference statistics from a pickle."""
    assert metric in ['fid', 'fd_dinov2'], f'Invalid metric name: {metric}.'
    with open(ref_path, 'rb') as f:
        data = pickle.load(f)
    return data[metric]['mu'], data[metric]['sigma']


def save_to_zip_file(images: np.ndarray,
                     zip_path: str) -> None:
    """Saves images to a zip file."""
    img_idx = 0
    with zipfile.ZipFile(zip_path, 'w') as f:
        for image in images:
            zip_fname = f'{img_idx:06d}.png'
            im = PIL.Image.fromarray(image.transpose(1, 2, 0))
            image_file = BytesIO()
            im.save(image_file, 'PNG')
            f.writestr(zip_fname, image_file.getvalue())
            img_idx += 1


def draw_box(img: np.ndarray,
             x: int,
             y: int,
             s: int,
             thick: float,
             color: List[int]):
    """Draws a red box to an image."""
    if thick <= 0:
        return img
    img = img.copy()
    for t in range(max(int(img.shape[0] * thick + 0.5), 1)):
        img[y + t, x : x + s] = color
        img[y - t + s - 1, x : x + s] = color
        img[y : y + s, x + t] = color
        img[y : y + s, x - t + s - 1] = color
    return img


def create_grid(grid_path: str,
                images: np.ndarray,
                num_cols: int,
                num_rows: int,
                inset: Optional[list] = None,
                thick_a: float = 7e-3,
                thick_b: float = 4e-3) -> None:
    """Creates and saves an image grid."""
    assert images.ndim == 4

    _, _, h, w = images.shape
    bg = PIL.Image.new(mode='RGB', size=(num_cols * w, num_rows * h), color=(0, 0, 0))
    for i in range(num_rows):
        for j in range(num_cols):
            if inset is None:  # Constant G grid.
                im = PIL.Image.fromarray(images[i * num_cols + j].transpose(1, 2, 0))
                bg.paste(im, box=(j * w, i * h))
            else:
                x, y, s, c = inset
                if i == num_rows - 1:  # Draw inset row.
                    img = images[(i - 1) * num_cols + j].transpose(1, 2, 0)
                    im = PIL.Image.fromarray(img).crop((x, y, x + s, y + s))
                    inset_img = np.uint8(im.resize((img.shape[1], img.shape[0]), PIL.Image.Resampling.LANCZOS))
                    inset_img = draw_box(inset_img, x=0, y=0, s=img.shape[0], thick=thick_b, color=c)
                    bg.paste(PIL.Image.fromarray(inset_img), box=(j * w, i * h))
                else:  # Draw inset in the original image.
                    img = images[i * num_cols + j].transpose(1, 2, 0)
                    img = draw_box(img, x=x, y=y, s=s, thick=thick_a, color=c)
                    bg.paste(PIL.Image.fromarray(img), box=(j * w, i * h))
    bg.save(grid_path)


def write_jsonl(path: str,
                data: dict) -> None:
    """Writes to JSONL file."""
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')
