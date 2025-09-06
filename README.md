## Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models</sub>

<figure>
    <p align="center">
        <img width="800" src="teaser.png">
        <br>
    </p>
</figure>

**Applying Guidance in a Limited Interval Improves
Sample and Distribution Quality in Diffusion Models**<br>
Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, Jaakko Lehtinen<br>
[https://arxiv.org/abs/2404.07724](https://arxiv.org/abs/2404.07724)<br>

Abstract: _Guidance is a crucial technique for extracting the best performance out of imagegenerating diffusion models. Traditionally, a constant guidance weight has been applied throughout the sampling chain of an image. We show that guidance is clearly harmful toward the beginning of the chain (high noise levels), largely unnecessary toward the end (low noise levels), and only beneficial in the middle. We thus restrict it to a specific range of noise levels, improving both the inference speed and result quality. This limited guidance interval improves the record FID in ImageNet-512 significantly, from 1.81 to 1.40. We show that it is quantitatively and qualitatively beneficial across different sampler parameters, network architectures, and datasets, including the large-scale setting of Stable Diffusion XL. We thus suggest exposing the guidance interval as a hyperparameter in all diffusion models that use guidance._

## Setup

We recommend using [Anaconda](https://www.anaconda.com/). To create a virtual environment and install required packages run:

```
conda env create -f environment.yml
conda activate guidance-interval
```

## Usage

This repository provides code to reproduce qualitative and quantitative ImageNet-512 results with EDM2-XXL model. The models used in this paper are available [here](https://drive.google.com/drive/folders/1nrxg-0LkAePD9HOWotS6q6E9MxpJelNv?usp=sharing).

### Qualitative results

All qualitative results from ImageNet-512 with EDM2-XXL can be reproduced with the following command:

```
python generate_images.py \\
    --cond_pkl=<path-to-cond-xxl-pickle> \\
    --uncond_pkl=<path-to-uncond-xs-pickle> \\
    --figs=all
```

See `--help` of `generate_images.py` for more instructions. The reference qualitative images can be found [here](https://drive.google.com/drive/folders/1nrxg-0LkAePD9HOWotS6q6E9MxpJelNv?usp=sharing).

### Quantitative results

First download pre-computed reference statistics from: https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/. Then, quantitative FID results from ImageNet-512 with EDM2-XXL can be reproduced with the following command:

```
# EDM2-XXL FID optimal results with guidance interval:
python generate_zip.py \\
    --cond_pkl=<path-to-cond-xxl-pickle> \\
    --uncond_pkl=<path-to-uncond-xs-pickle> \\
    --zip_dir=<dir-where-zip-is-saved> \\
    --guidance_interval="[17, 22]" \\
    --guidance_scale=2.0

# Evaluate FID.
python calculate_metrics.py \\
    --gen_zip=<path-to-gen-zip> \\
    --ref_path=<path-to-img512.pkl>
```

See `--help` of `generate_images.py` and `calculate_metrics.py` for more instructions.

## License

The code of this repository is released under the Apache License 2.0.

This repository adapts code from public repositories: [StyleGAN2](https://github.com/NVlabs/stylegan2), [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) and [EDM](https://github.com/NVlabs/edm), which are released under Nvidia Source Code License-NC, Nvidia Source Code License-NC, and Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, respectively.

The pre-trained models are from [EDM2](https://github.com/NVlabs/edm2) and they are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## Acknowledgements

This work was partially supported by the European Research Council (ERC Consolidator Grant 866435), and made use of computational resources provided by the Aalto Science-IT project and the Finnish IT Center for Science (CSC).

## Citation

```
@inproceedings{Kynkaanniemi2024,
    author = {Tuomas Kynkäänniemi and
              Miika Aittala and
              Tero Karras and
              Samuli Laine and
              Timo Aila and
              Jaakko Lehtinen},
    title = {Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models},
    booktitle = {Proc. NeurIPS},
    year = {2024},
}
```
