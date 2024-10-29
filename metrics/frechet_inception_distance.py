# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Fréchet Inception Distance (FID). Adapted from Karras et al.:
   https://github.com/NVlabs/stylegan2/blob/master/metrics/frechet_inception_distance.py"""

import numpy as np
import scipy.linalg


def compute_fid(gen_features: np.ndarray,
                mu_ref: np.ndarray,
                sigma_ref: np.ndarray) -> float:
    """Computes the Fréchet Inception Distance."""
    assert gen_features.ndim == 2

    # Feature statistics.
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # FID.
    assert mu_ref.shape == mu_gen.shape
    assert sigma_ref.shape == sigma_gen.shape
    m = np.square(mu_gen - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_ref), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_ref - s * 2))

    return fid
