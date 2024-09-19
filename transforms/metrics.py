from typing import Tuple

import torch
import numpy as np
from torch import Tensor
from pytorch_msssim import ssim as pt_ssim
from pytorch_msssim import ms_ssim as pt_ms_ssim

# Scikit-image version for CPU:
#
# from numpy.typing import ArrayLike
# from skimage.metrics import (
#     peak_signal_noise_ratio,
#     mean_squared_error,
#     structural_similarity,
# )

# All metrics assume pixel values between 0.0 and 1.0.


# (C,H,W) -> (mse, psnr)
def mse_psnr(img: Tensor, img_ref: Tensor) -> Tuple[float, float]:
    mse_val = mse(img, img_ref)
    return (mse_val.item(), (10.0 * torch.log10(1.0 / mse_val)).item())


# (N,C,H,W) -> N
def ssim(batch: Tensor, batch_ref: Tensor) -> Tensor:
    return pt_ssim(batch, batch_ref, data_range=1.0, size_average=False)


# (N,C,H,W) -> N
def msssim(batch: Tensor, batch_ref: Tensor) -> Tensor:
    return pt_ms_ssim(batch, batch_ref, data_range=1.0, size_average=False)


# (C,H,W) -> scalar tensor
def mse(img: Tensor, img_ref: Tensor) -> Tensor:
    return (img - img_ref).square().mean()

# MSE in DCT domain
def mse_dct(img: Tensor, img_ref: Tensor) -> Tensor:
    pass

# Scikit-image version for CPU:

# def mse(img: ArrayLike, img_ref: ArrayLike) -> np.float64:
#     return mean_squared_error(img_ref, img)

# def psnr(img: ArrayLike, img_ref: ArrayLike) -> np.float64:
# return peak_signal_noise_ratio(img_ref, img)

# def ssim(img: ArrayLike, img_ref: ArrayLike) -> np.float64:
# return structural_similarity(
#     img_ref, img, data_range=img.max() - img.min(), channel_axis=0
# )
