# Adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/color/ycbcr.html#rgb_to_ycbcr
#
# w_human = (0.299, 0.587, 0.114)

import torch
import torch.nn as nn

from torch import Tensor

def _rgb_to_y(r: Tensor, g: Tensor, b: Tensor, w: Tensor) -> Tensor:
    y: Tensor = w[0] * r + w[1] * g + w[2] * b
    return y


def rgb_to_y(image: Tensor) -> Tensor:
    r"""Convert an RGB image to Y.

    Args:
        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    y: Tensor = _rgb_to_y(r, g, b)
    return y


def rgb_to_ycbcr(image: Tensor, w: Tensor) -> Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> w_human = (0.299, 0.587, 0.114)
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input, w_human)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    wr, wg, wb = w

    delta: float = 0.5
    y: Tensor = _rgb_to_y(r, g, b, w)
    cb: Tensor = (b - y) * (0.5 / (1 - wb)) + delta #0.564 + delta
    cr: Tensor = (r - y) * (0.5 / (1 - wr)) + delta #0.713 + delta
    return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: Tensor, w: Tensor) -> Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: Tensor = image[..., 0, :, :]
    cb: Tensor = image[..., 1, :, :]
    cr: Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: Tensor = cb - delta
    cr_shifted: Tensor = cr - delta

    wr, wg, wb = w

    r: Tensor = y + (2 - 2 * wr) * cr_shifted
    g: Tensor = y - (wr / wg * (2 - 2 * wr)) * cr_shifted - (wb / wg * (2 - 2 * wb)) * cb_shifted
    b: Tensor = y + (2 - 2 * wb) * cb_shifted
    return torch.stack([r, g, b], -3)


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> w_human = (0.299, 0.587, 0.114)
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr(w_human)
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def __init__(self, w: Tensor):
        super().__init__()
        self.w = w

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_ycbcr(image, self.w)


class YcbcrToRgb(nn.Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> w_human = (0.299, 0.587, 0.114)
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb(w_human)
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self, w: Tensor):
        super().__init__()
        self.w = w

    def forward(self, image: Tensor) -> Tensor:
        return ycbcr_to_rgb(image, self.w)
