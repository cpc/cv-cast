import itertools
import math
import os
import sys
from typing import Callable, TypedDict, Literal

import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, List
from pathlib import Path
from packaging import version

from torch import Tensor
from torch.nn import ReplicationPad2d

import torchvision
from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

# from IPython.core.debugger import set_trace

import scipy.io as sio

# Required for JPEG:
import cv2 as cv
from turbojpeg import TurboJPEG, TJSAMP_420

import torchjpeg.codec
from gradients.grace import get_quant_table_approx

from .icm.icm import ICM
from .icm.utils import pad as icm_pad
from .icm.utils import crop as icm_crop
from .tcm.tcm import TCM

DEFAULT_JPEG_LIB = "/home/jakub/git/cpc/aisa-demo/external/libjpeg-turbo/libturbojpeg/x86-64/lib/libturbojpeg.so"

# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["savefig.dpi"] = 300

# Defines how the DCT coefficients are grouped into chunks in the case of chroma
# subsampling (subsampled chunks have less coefficients and we need to group them to
# make all chunks have the same number of coefficients).
ALLOWED_DCT_GROUPING = {
    # four subsampled DCT chunks are grouped into one
    420: [
        #   U U
        #   V V
        "horizontal_uv",
        #   U V
        #   U V
        "vertical_uv",
    ],
    # no subsampling => no grouping happens
    444: ["horizontal_uv", "vertical_uv"],
}


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


# import hack to make this work with fastseg
if versiontuple(sys.version.split()[0]) < versiontuple("3.9.0"):
    from ..transforms.dct import dct_2d, idct_2d
    from ..transforms.color_transforms import RgbToYcbcr, YcbcrToRgb
    from ..utils import set_device
else:
    from transforms.dct import dct_2d, idct_2d
    from transforms.color_transforms import RgbToYcbcr, YcbcrToRgb
    from utils import set_device


class YuvImage:
    """Image holding YUV channels separately so they can be subsampled."""

    def __init__(self, y: Tensor, u: Tensor, v: Tensor):
        self.y = y.squeeze()
        self.u = u.squeeze()
        self.v = v.squeeze()

    def __repr__(self):
        return "YuvImage(y={}, u={}, v={})".format(
            self.y.shape, self.u.shape, self.v.shape
        )

    @property
    def shape(self):
        return (3, self.y.shape[-2], self.y.shape[-1])

    def channels(self) -> List[Tensor]:
        return [self.y, self.u, self.v]

    def mean(self) -> Tensor:
        return torch.tensor([self.y.mean(), self.u.mean(), self.v.mean()])

    def sub(self, other: Tensor):
        return YuvImage(
            self.y - other[0, None, None],
            self.u - other[1, None, None],
            self.v - other[2, None, None],
        )

    def diff(self, other):
        return YuvImage(self.y - other.y, self.u - other.u, self.v - other.v)

    def add(self, other: Tensor):
        return YuvImage(
            self.y + other[0, None, None],
            self.u + other[1, None, None],
            self.v + other[2, None, None],
        )

    def square(self):
        return YuvImage(self.y.square(), self.u.square(), self.v.square())

    def abs(self):
        return YuvImage(self.y.abs(), self.u.abs(), self.v.abs())

    def min(self):
        mins = torch.Tensor([self.y.min(), self.u.min(), self.v.min()])
        return torch.min(mins)

    def max(self):
        maxs = torch.Tensor([self.y.max(), self.u.max(), self.v.max()])
        return torch.max(maxs)

    # def get_dc(self) -> Tensor:
    #     return torch.Tensor([self.y[0, 0].item(), self.u[0, 0].item(), self.v[0, 0].item()])

    def to(self, device: torch.device, non_blocking: bool = True):
        self.y = self.y.to(device, non_blocking=non_blocking)
        self.u = self.u.to(device, non_blocking=non_blocking)
        self.v = self.v.to(device, non_blocking=non_blocking)

        return self


class Metadata:
    """LVC metadata"""

    def __init__(
        self,
        image_size: Tuple[int, int],
        chunk_size: Tuple[int, int],
        nchunks: Tuple[int, int],
    ):
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.nchunks = nchunks
        self.padded_size = image_size
        self.noise_power = 0.0

    def set_variances(self, variances: Tensor):
        self.variances = variances.detach().clone()

    def set_all_variances(self, all_variances: Tensor):
        self.all_variances = all_variances.detach().clone()

    def set_bitmap(self, bitmap: Tensor):
        self.bitmap = bitmap.detach().clone()

    def set_mean(self, mean: Tensor):
        self.mean = mean.detach().clone()

    def set_padded_size(self, padded_size: Tuple[int, int]):
        self.padded_size = padded_size

    def set_noise_power(self, noise_power: Tensor):
        self.noise_power = noise_power.detach().clone()

    def print_shapes(self):
        print(
            "Metadata shapes: mean: {}, variances: {}, bitmap: {}".format(
                self.mean.shape, self.variances.shape, self.bitmap.shape
            )
        )

    def to(self, device: torch.device, non_blocking: bool = True):
        try:
            self.variances = self.variances.to(device, non_blocking=non_blocking)
        except:
            pass

        try:
            self.all_variances = self.all_variances.to(
                device, non_blocking=non_blocking
            )
        except:
            pass

        try:
            self.bitmap = self.bitmap.to(device, non_blocking=non_blocking)
        except:
            pass

        try:
            self.mean = self.mean.to(device, non_blocking=non_blocking)
        except:
            pass

        try:
            self.noise_power = self.noise_power.to(device, non_blocking=non_blocking)
        except:
            pass

        return self


def get_chunk_size(
    chunk_size: Tuple[int, int] | int, img_h: int, img_w: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if isinstance(chunk_size, tuple):
        chunk_h, chunk_w = chunk_size
        nchunks_x = math.ceil(img_w / chunk_w)
        nchunks_y = math.ceil(img_h / chunk_h)
    else:
        nchunks = chunk_size
        nchunks_xy = round(math.sqrt(nchunks))
        chunk_w = math.ceil(img_w / nchunks_xy)
        chunk_h = math.ceil(img_h / nchunks_xy)
        nchunks_x = nchunks_xy
        nchunks_y = nchunks_xy

    return ((chunk_h, chunk_w), (nchunks_y, nchunks_x))


def get_padded_size(
    chunk_size: Tuple[int, int],
    nchunks: Tuple[int, int],
    dct_size: Tuple[int, int] | None,
    img_h: int,
    img_w: int,
    mode: int,
) -> Tuple[int, int]:
    chunk_h, chunk_w = chunk_size
    orig_chunk_h, orig_chunk_w = chunk_size
    nchunks_y, nchunks_x = nchunks

    if mode == 420:
        chunk_w = 2 * chunk_w
        chunk_h = 2 * chunk_h

    if dct_size is not None:
        dct_h, dct_w = dct_size

        if mode == 420:
            dct_w = 2 * dct_w
            dct_h = 2 * dct_h

        chunk_w = chunk_w * dct_w / math.gcd(chunk_w, dct_w)
        chunk_h = chunk_h * dct_h / math.gcd(chunk_h, dct_h)

    padded_w = int(math.ceil(img_w / nchunks_x) * nchunks_x)
    padded_h = int(math.ceil(img_h / nchunks_y) * nchunks_y)

    if dct_size is not None:
        dct_h, dct_w = dct_size

        if (dct_w * dct_h) != (nchunks_x * nchunks_y):
            raise ValueError(
                "Padding Error: DCT size {}x{} results in {} chunks but the number of {}x{} chunks is {}.".format(
                    dct_w, dct_h, dct_w * dct_h, orig_chunk_w, orig_chunk_h, nchunks
                )
            )

    return (padded_h, padded_w)


class WrapMetadata(nn.Module):
    """Wrap input tensor with dummy metadata

    Intended for compatibility between transforms that accept and do not accept
    metadata input.
    """

    def __init__(self, chunk_size: Tuple[int, int] | int):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, inp: Tensor) -> Tuple[Tensor, Metadata]:
        try:
            image_size = (inp.shape[-2], inp.shape[-1])
        except TypeError:
            image_size = (inp.shape()[-2], inp.shape()[-1])

        chunk_size, nchunks = get_chunk_size(
            self.chunk_size, image_size[0], image_size[1]
        )

        metadata = Metadata(image_size, chunk_size, nchunks)

        return (inp, metadata)


class StripMetadata(nn.Module):
    """Strip metadata from input

    Intended for compatibility between transforms that accept and do not accept
    metadata input.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp: Tensor, metadata: Metadata) -> Tensor:
        return inp


class ToDevice(nn.Module):
    """Move tensor to device"""

    def __init__(self, device: str | torch.device = "cpu", non_blocking: bool = True):
        super().__init__()
        self.device = device
        self.non_blocking = non_blocking

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        return (
            inp.to(self.device, non_blocking=self.non_blocking),
            metadata.to(self.device, non_blocking=self.non_blocking),
        )


class Pad(nn.Module):
    """Calculate DCT of an image"""

    def __init__(
        self, mode: int, dct_size: Tuple[int, int] | None, do_print: bool = False
    ):
        super().__init__()
        self.mode = mode
        self.dct_size = dct_size
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        img_h, img_w = (inp.shape[-2], inp.shape[-1])
        chunk_h, chunk_w = metadata.chunk_size

        padded_h, padded_w = get_padded_size(
            metadata.chunk_size,
            metadata.nchunks,
            self.dct_size,
            img_h,
            img_w,
            self.mode,
        )

        if len(inp.shape) == 3:
            inp = inp.unsqueeze(0)

        res = inp

        if self.mode == 420:
            chunk_w = 2 * chunk_w
            chunk_h = 2 * chunk_h

        if (img_w != padded_w) or (img_h != padded_h):
            w_pad = padded_w - img_w
            h_pad = padded_h - img_h
            padding = nn.ReplicationPad2d((0, w_pad, 0, h_pad))
            res = padding(inp)

            if self.do_print:
                print(
                    "Padded image: {}x{} -> {}x{}".format(
                        img_w, img_h, padded_w, padded_h
                    )
                )
        else:
            if self.do_print:
                print("Padded image: {}x{} (no padding)".format(img_w, img_h))

        metadata.set_padded_size((padded_h, padded_w))

        return (res.squeeze(), metadata)


class Crop(nn.Module):
    """Calculate DCT of an image"""

    def __init__(self, do_print: bool = False):
        super().__init__()
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        img_h, img_w = metadata.image_size
        padded_h, padded_w = metadata.padded_size

        if version.parse(torchvision.__version__) <= version.parse("0.2.2"):
            inp = transforms.ToPILImage()(inp)

        if (img_w != padded_w) or (img_h != padded_h):
            res = transforms.functional.crop(inp, 0, 0, img_h, img_w)
            if self.do_print:
                print(
                    "Cropped image: {}x{} -> {}x{}".format(
                        padded_w, padded_h, img_w, img_h
                    )
                )
        else:
            res = inp
            if self.do_print:
                print("Cropped image: {}x{} (no cropping)".format(img_w, img_h))

        if version.parse(torchvision.__version__) <= version.parse("0.2.2"):
            res = transforms.ToTensor()(res)

        return (res, metadata)


class DctYuv(nn.Module):
    def __init__(self, is_half: bool, norm: str = "ortho"):
        super().__init__()
        self.is_half = is_half
        self.norm = norm

    def forward(self, inp: YuvImage) -> YuvImage:
        if self.is_half:
            res = YuvImage(
                dct_2d(inp.y.float(), norm=self.norm).squeeze().half(),
                dct_2d(inp.u.float(), norm=self.norm).squeeze().half(),
                dct_2d(inp.v.float(), norm=self.norm).squeeze().half(),
            )
        else:
            res = YuvImage(
                dct_2d(inp.y, norm=self.norm).squeeze(),
                dct_2d(inp.u, norm=self.norm).squeeze(),
                dct_2d(inp.v, norm=self.norm).squeeze(),
            )

        return res


class IdctYuv(nn.Module):
    def __init__(self, is_half: bool, norm: str = "ortho"):
        super().__init__()
        self.is_half = is_half
        self.norm = norm

    def forward(self, inp: YuvImage) -> YuvImage:
        if self.is_half:
            res = YuvImage(
                idct_2d(inp.y.float(), norm=self.norm).squeeze().half(),
                idct_2d(inp.u.float(), norm=self.norm).squeeze().half(),
                idct_2d(inp.v.float(), norm=self.norm).squeeze().half(),
            )
        else:
            res = YuvImage(
                idct_2d(inp.y, norm=self.norm).squeeze(),
                idct_2d(inp.u, norm=self.norm).squeeze(),
                idct_2d(inp.v, norm=self.norm).squeeze(),
            )

        return res


class DctYuvMetadata(nn.Module):
    """Calculate DCT of an image with metadata"""

    def __init__(self, is_half: bool, norm: str = "ortho", results: List | None = None):
        super().__init__()
        self.results = results
        self.dct = DctYuv(is_half, norm)

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        res = self.dct(inp)

        if self.results is not None:
            self.results.append(res)

        return (res, metadata)


class IdctYuvMetadata(nn.Module):
    """Calculate inverse DCT of an image"""

    def __init__(self, is_half: bool, norm: str = "ortho"):
        super().__init__()
        self.idct = IdctYuv(is_half, norm)

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        res = self.idct(inp)

        return (res, metadata)


class DctBlock(nn.Module):
    """Calculate DCT performed on blocks of pixels

    It takes the input image already split into blocks equal to the DCT size.
    After the DCT, it rearranges the DCT coefficients into chunks used for LVC.
    """

    def __init__(
        self,
        dct_size: Tuple[int, int],
        mode: int,
        grouping: str,
        is_half: bool,
        norm=None,
        do_print=False,
        results: List = None,
    ):
        if grouping not in ALLOWED_DCT_GROUPING[mode]:
            raise ValueError(
                "DctBlock: Invalid grouping '{}'. Choose from {}".format(
                    grouping, ALLOWED_DCT_GROUPING
                )
            )

        if (dct_size[0] % 2 != 0) or (dct_size[1] % 2 != 0):
            raise ValueError(
                "DctBlock: Invalid DCT block size {}. Must be divisible by 2.".format(
                    dct_size
                )
            )

        super().__init__()
        self.dct_size = dct_size
        self.mode = mode
        self.grouping = grouping
        self.is_half = is_half
        self.norm = norm
        self.do_print = do_print
        self.results = results
        self.superprint = False

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        if self.do_print:
            print("DCT block: inp shape: ", inp.shape)

        if self.is_half:
            res = dct_2d(inp.float(), norm=self.norm).half()
        else:
            res = dct_2d(inp, norm=self.norm)

        if self.dct_size is None:
            raise ValueError("DCTBlock: DCT size must be set. It is None.")

        chunk_h, chunk_w = metadata.chunk_size
        dct_h, dct_w = self.dct_size
        padded_h, padded_w = metadata.padded_size

        nblocks_y = int((padded_w / dct_w) * (padded_h / dct_h))
        if self.mode == 420:
            nblocks_uv = int(nblocks_y / 4)
        else:
            nblocks_uv = nblocks_y

        blocks_y = res[:nblocks_y]
        blocks_u = res[nblocks_y : nblocks_y + nblocks_uv]
        blocks_v = res[nblocks_y + nblocks_uv :]
        if self.do_print:
            print("DCT padded size: {}x{}".format(padded_w, padded_h))
            print("DCT nblocks YUV: ", nblocks_y, nblocks_uv, nblocks_uv)
            print("DCT block_y size: ", blocks_y.shape)
            print("DCT block_u size: ", blocks_u.shape)
            print("DCT block_v size: ", blocks_v.shape)

        if self.mode == 420:
            if self.grouping == "horizontal_uv":
                split_dim = 2
            elif self.grouping == "vertical_uv":
                split_dim = 1
            else:
                raise ValueError(
                    "DctBlock: Invalid grouping '{}'. Choose from {}".format(
                        self.grouping, ALLOWED_DCT_GROUPING
                    )
                )

            blocks = []
            for bu, bv in zip(
                blocks_u.split(2, dim=split_dim), blocks_v.split(2, dim=split_dim)
            ):
                b = torch.cat([bu, bv], dim=split_dim)
                blocks.append(b)

            if self.do_print:
                print("DCT block_uv orig size: ", torch.cat(blocks, dim=0).shape)
                if self.superprint:
                    print("DCT block orig [0]:")
                    print(blocks[0].shape)

            blocks_uv = (
                torch.cat(blocks, dim=0).permute(1, 2, 0).reshape(-1, chunk_h, chunk_w)
            )

            if self.do_print:
                print("DCT block_uv final size: ", blocks_uv.shape)
        else:
            blocks_u = blocks_u.permute(1, 2, 0).reshape(-1, chunk_h, chunk_w)
            blocks_v = blocks_v.permute(1, 2, 0).reshape(-1, chunk_h, chunk_w)
            blocks_uv = torch.cat([blocks_u, blocks_v], dim=0)

        if self.results is not None:
            chunks_y = blocks_y.permute(1, 2, 0).reshape(-1, chunk_h, chunk_w)
            chunks_u = blocks_u.permute(1, 2, 0).reshape(-1, chunk_h, chunk_w)
            chunks_v = blocks_v.permute(1, 2, 0).reshape(-1, chunk_h, chunk_w)
            self.results.append(
                {
                    "Y": chunks_y.var(dim=(1, 2)),
                    "U": chunks_u.var(dim=(1, 2)),
                    "V": chunks_v.var(dim=(1, 2)),
                }
            )

        blocks_y = blocks_y.permute(1, 2, 0).reshape(-1, chunk_h, chunk_w)
        res = torch.cat([blocks_y, blocks_uv], dim=0)

        if self.do_print:
            print("DCT result size: ", res.shape)

        if self.is_half:
            res = res.half()

        return (res, metadata)


class IdctBlock(nn.Module):
    """Calculate inverse DCT performed on blocks of coefficients

    It takes LVC chunks as an input, rearranges them into DCT blocks and runs IDCT on
    them. Returns the original image split into chunks equal to the DCT size.
    """

    def __init__(
        self,
        dct_size: Tuple[int, int],
        mode: int,
        grouping: str,
        is_half: bool,
        norm=None,
        do_print=False,
    ):
        if grouping not in ALLOWED_DCT_GROUPING[mode]:
            raise ValueError(
                "DctBlock: Invalid grouping '{}'. Choose from {}".format(
                    grouping, ALLOWED_DCT_GROUPING
                )
            )

        super().__init__()
        self.dct_size = dct_size
        self.mode = mode
        self.grouping = grouping
        self.is_half = is_half
        self.norm = norm
        self.do_print = do_print
        self.superprint = False

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        if self.dct_size is None:
            raise ValueError("IDCTBlock: DCT size must be set. It is None.")

        if self.do_print:
            print("IDCT inp shape: ", inp.shape)

        chunk_h, chunk_w = metadata.chunk_size
        dct_h, dct_w = self.dct_size
        padded_h, padded_w = metadata.padded_size

        nblocks_y = int((padded_w / chunk_w) * (padded_h / chunk_h))
        if self.mode == 420:
            nblocks_uv = int(nblocks_y / 4)
        else:
            nblocks_uv = nblocks_y

        if self.do_print:
            print("IDCT nchunks: ", nblocks_y, nblocks_uv, nblocks_uv)

        blocks_y = inp[:nblocks_y].permute(1, 2, 0).reshape(-1, dct_h, dct_w)
        blocks_uv = inp[nblocks_y:]

        if self.do_print:
            print("IDCT blocks_uv size: ", blocks_uv.shape)

        blocks_uv = blocks_uv.permute(1, 2, 0)

        if self.do_print:
            print("IDCT blocks_uv permuted size: ", blocks_uv.shape)
            if self.superprint:
                print("IDCT blocks_uv permuted:")
                print(blocks_uv[0].shape)

        if self.mode == 420:
            if self.grouping == "horizontal_uv":
                split_dim = 2
                blocks_uv = blocks_uv.reshape(-1, dct_h, 2 + 2)
                nsplits = dct_w // 2
            elif self.grouping == "vertical_uv":
                split_dim = 1
                blocks_uv = blocks_uv.reshape(-1, 2 + 2, dct_w)
                nsplits = dct_h // 2
            else:
                raise ValueError(
                    "DctBlock: Invalid grouping '{}'. Choose from {}".format(
                        self.grouping, ALLOWED_DCT_GROUPING
                    )
                )

            if self.do_print:
                print("IDCT blocks_uv reshaped size: ", blocks_uv.shape)

            blocks_u_splits = []
            blocks_v_splits = []
            for pairs_uv in blocks_uv.chunk(nsplits, dim=0):
                pairs_u, pairs_v = pairs_uv.split(2, dim=split_dim)
                blocks_u_splits.append(pairs_u)
                blocks_v_splits.append(pairs_v)

            blocks_u = torch.cat(blocks_u_splits, dim=split_dim)
            blocks_v = torch.cat(blocks_v_splits, dim=split_dim)

            if self.do_print:
                print("IDCT blocks_u/v split size: ", blocks_u.shape, blocks_v.shape)
        else:
            blocks_u = (
                inp[nblocks_y : nblocks_y + nblocks_uv]
                .permute(1, 2, 0)
                .reshape(-1, dct_h, dct_w)
            )
            blocks_v = (
                inp[nblocks_y + nblocks_uv :].permute(1, 2, 0).reshape(-1, dct_h, dct_w)
            )

        if self.do_print:
            print("IDCT block_y size: ", blocks_y.shape)
            print("IDCT block_y permute size: ", blocks_y.permute(1, 2, 0).shape)

        blocks = torch.cat([blocks_y, blocks_u, blocks_v], dim=0)

        if self.do_print:
            print("IDCT blocks final size: ", blocks.shape)

        if self.is_half:
            res = idct_2d(blocks.float(), norm=self.norm).half()
        else:
            res = idct_2d(blocks, norm=self.norm)

        if self.do_print:
            print("IDCT result size: ", res.shape)

        if self.is_half:
            res = res.half()

        return (res, metadata)


class TensorToImage(nn.Module):
    """Convert input image tensor into YuvImage class instance"""

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        return (YuvImage(inp[0, :, :], inp[1, :, :], inp[2, :, :]), metadata)


class ImageToTensor(nn.Module):
    """Convert YuvImage class instance into input image tensor"""

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        return (torch.stack([inp.y, inp.u, inp.v]), metadata)


class RgbToYcbcrMetadata(RgbToYcbcr):
    """RGB -> YCbCr transform, passing along metadata"""

    def __init__(self, w: Tensor):
        super().__init__(w)

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        ycbcr = super().forward(inp)

        return (YuvImage(ycbcr[0, :, :], ycbcr[1, :, :], ycbcr[2, :, :]), metadata)


class YcbcrToRgbMetadata(YcbcrToRgb):
    """RGB -> YCbCr transform, passing along metadata"""

    def __init__(self, w: Tensor, clamp=True):
        super().__init__(w)
        self.clamp = clamp

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        rgb = super().forward(torch.stack([inp.y, inp.u, inp.v]))

        if self.clamp:
            rgb = rgb.clamp(0.0, 1.0)

        return (rgb, metadata)


class DownsampleTensor(nn.Module):
    """Downsample tensor as padded tensor"""

    def __init__(self, mode: int, filter: str):
        """Create new instance of tensor downsampler.

        mode:    YUV subsampling mode (420 or 444)
        filter:  What filter to use for interpolation (nearest, bilinear, ...)
        """
        super().__init__()

        if mode != 444 and mode != 420:
            raise ValueError(
                "Invalid subsampling mode {} (choose 444 or 420)".format(mode)
            )

        self.mode = mode
        self.filter = filter

    def forward(self, inp: Tensor) -> Tensor:
        H = inp.shape[-2]
        W = inp.shape[-1]

        y, u, v = (x.view((1, 1, H, W)) for x in inp.unbind(-3))

        if self.mode == 420:
            u = nn.functional.interpolate(u, scale_factor=0.5, mode=self.filter)
            pad_w = W - u.shape[-1]
            pad_h = H - u.shape[-2]
            pad = nn.ConstantPad2d((0, pad_w, 0, pad_h), 0.0)  # lrtb
            u = pad(u)

            v = nn.functional.interpolate(v, scale_factor=0.5, mode=self.filter)
            pad_w = W - v.shape[-1]
            pad_h = H - v.shape[-2]
            pad = nn.ConstantPad2d((0, pad_w, 0, pad_h), 0.0)  # lrtb
            v = pad(v)

            return torch.cat([y, u, v], dim=1)
        else:
            return inp


class UpsampleTensor(nn.Module):
    """Downsample padded tensor as tensor"""

    def __init__(self, mode=444):
        super().__init__()

        if mode != 444 and mode != 420:
            raise ValueError(
                "Invalid subsampling mode {} (choose 444 or 420)".format(mode)
            )

        self.mode = mode

    def forward(self, inp: Tensor) -> Tensor:
        H = inp.shape[-2]
        W = inp.shape[-1]

        y, u, v = (x.view((1, 1, H, W)) for x in inp.unbind(-3))

        if self.mode == 420:
            sub_w = int(W / 2)
            sub_h = int(H / 2)

            u = u[:, :, :sub_h, :sub_w]
            v = v[:, :, :sub_h, :sub_w]

            u = nn.functional.interpolate(u, scale_factor=2.0, mode="bilinear")
            v = nn.functional.interpolate(v, scale_factor=2.0, mode="bilinear")

            return torch.cat([y, u, v], dim=1)
        else:
            return inp


class DownsampleTensorOld(nn.Module):
    """Downsample tensor as YUV image"""

    def __init__(self, mode=444):
        super().__init__()

        if mode != 444 and mode != 420:
            raise ValueError(
                "Invalid subsampling mode {} (choose 444 or 420)".format(mode)
            )

        self.mode = mode

    def forward(self, inp: Tensor) -> YuvImage:
        H = inp.shape[-2]
        W = inp.shape[-1]

        y, u, v = (x.view((1, 1, H, W)) for x in inp.unbind(-3))

        # y = inp.y.view((1, 1, H, W))
        # u = inp.u.view((1, 1, H, W))
        # v = inp.v.view((1, 1, H, W))

        if self.mode == 420:
            u = nn.functional.interpolate(u, scale_factor=0.5, mode="bilinear")
            v = nn.functional.interpolate(v, scale_factor=0.5, mode="bilinear")

        return YuvImage(y.squeeze(), u.squeeze(), v.squeeze())


class Downsample(nn.Module):
    """Downsample YUV image"""

    def __init__(self, mode=444):
        super().__init__()

        if mode != 444 and mode != 420:
            raise ValueError(
                "Invalid subsampling mode {} (choose 444 or 420)".format(mode)
            )

        self.mode = mode

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        _, H, W = inp.shape

        y = inp.y.view((1, 1, H, W))
        u = inp.u.view((1, 1, H, W))
        v = inp.v.view((1, 1, H, W))

        if self.mode == 420:
            u = nn.functional.interpolate(u, scale_factor=0.5, mode="bilinear")
            v = nn.functional.interpolate(v, scale_factor=0.5, mode="bilinear")

        return (YuvImage(y.squeeze(), u.squeeze(), v.squeeze()), metadata)


class Upsample(nn.Module):
    """Upsample YUV image to original resolution"""

    def __init__(self, mode=444):
        super().__init__()

        if mode != 444 and mode != 420:
            raise ValueError(
                "Invalid subsampling mode {} (choose 444 or 420)".format(mode)
            )

        self.mode = mode

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        y = inp.y.view(1, 1, inp.y.shape[-2], inp.y.shape[-1])
        u = inp.u.view(1, 1, inp.u.shape[-2], inp.u.shape[-1])
        v = inp.v.view(1, 1, inp.v.shape[-2], inp.v.shape[-1])

        if self.mode == 420:
            u = nn.functional.interpolate(u, scale_factor=2.0, mode="bilinear")
            v = nn.functional.interpolate(v, scale_factor=2.0, mode="bilinear")

        return (YuvImage(y.squeeze(), u.squeeze(), v.squeeze()), metadata)


class SubtractMean(nn.Module):
    """Subtract mean from the input samples and store it in metadata"""

    def __init__(self, device: torch.device, results: List = None):
        super().__init__()

        self.device = device
        self.results = results

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        inp = inp.to(self.device)
        metadata.mean = inp.mean().to(self.device, non_blocking=True)

        if self.results is not None:
            self.results.append(metadata.mean)

        return (inp.sub(metadata.mean[:, None, None]), metadata)


class RestoreMean(nn.Module):
    """Subtract mean from the input samples and store it in metadata"""

    def __init__(self):
        super().__init__()

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        return (inp.add(metadata.mean), metadata)


class ChunkSelect(nn.Module):
    """Select which chunks will be transmitted"""

    def __init__(self, cr: float, results: List = None, do_print: bool = False):
        super().__init__()

        self.cr = cr
        self.results = results
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        chunks = inp

        # Extract variance of each chunk
        variances = chunks.var(dim=(1, 2))

        if self.results is not None:
            self.results.append(variances)

        # Figure out which chunks to send
        ntotal = chunks.shape[0]
        nsend = math.floor(ntotal * self.cr)
        if self.do_print:
            print("total chunks: {}, to send: {}".format(ntotal, nsend))

        _, top_chunks_indices = variances.topk(nsend)
        top_chunks_indices, _ = top_chunks_indices.sort()
        top_chunks = chunks[top_chunks_indices]

        bitmap = torch.zeros_like(variances, dtype=torch.bool)
        bitmap[top_chunks_indices] = True

        metadata.set_bitmap(bitmap)
        metadata.set_variances(variances[bitmap])
        metadata.set_all_variances(variances)

        # Print tensor shapes as a sanity check
        if self.do_print:
            print("Sent chunks :", top_chunks.shape)
            print("Sent indices:", top_chunks_indices)
            metadata.print_shapes()

        return (top_chunks, metadata)


class ChunkSelectG(nn.Module):
    """Select which chunks will be transmitted, according to NN gradients"""

    def __init__(
        self, cr: float, g_norm: Tensor, results: List = None, do_print: bool = False
    ):
        super().__init__()

        self.cr = cr
        self.g_norm = g_norm.reshape(-1)
        self.results = results
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        chunks = inp

        # Extract variance of each chunk
        variances = chunks.var(dim=(1, 2))
        g_sq = self.g_norm.square()

        if self.results is not None:
            self.results.append(variances)

        # Figure out which chunks to send
        ntotal = chunks.shape[0]
        nsend = math.floor(ntotal * self.cr)
        if self.do_print:
            print("total chunks: {}, to send: {}".format(ntotal, nsend))

        g_variances = g_sq * variances

        _, top_chunks_indices = g_variances.topk(nsend)
        top_chunks_indices, _ = top_chunks_indices.sort()
        top_chunks = chunks[top_chunks_indices]

        bitmap = torch.zeros_like(variances, dtype=torch.bool)
        bitmap[top_chunks_indices] = True

        metadata.set_bitmap(bitmap)
        metadata.set_variances(variances[bitmap])
        metadata.set_all_variances(variances)

        # Print tensor shapes as a sanity check
        if self.do_print:
            print("Sent chunks :", top_chunks.shape)
            print("Sent indices:", top_chunks_indices)
            metadata.print_shapes()

        return (top_chunks, metadata)


class ChunkSelectGPrecise(nn.Module):
    """Select which chunks will be transmitted, according to NN gradients

    More precise method
    """

    def __init__(
        self, cr: float, g_mean: Tensor, results: List = None, do_print: bool = False
    ):
        super().__init__()

        self.cr = cr
        self.g_mean = g_mean
        self.results = results
        self.chunk_split = ChunkSplit(None, do_print=do_print)
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        chunks = inp
        g_chunks = self.chunk_split(
            YuvImage(
                self.g_mean[0],
                self.g_mean[1],
                self.g_mean[2],
            ),
            metadata,
        )[0]

        gx = (g_chunks[0] * chunks).square().sum(dim=(1, 2))

        # Figure out which chunks to send
        ntotal = chunks.shape[0]
        nsend = math.floor(ntotal * self.cr)
        if self.do_print:
            print("total chunks: {}, to send: {}".format(ntotal, nsend))

        _, top_chunks_indices = gx.topk(nsend)
        top_chunks_indices, _ = top_chunks_indices.sort()
        top_chunks = chunks[top_chunks_indices]

        bitmap = torch.zeros_like(gx, dtype=torch.bool)
        bitmap[top_chunks_indices] = True

        # Extract variance of each chunk
        variances = chunks.var(dim=(1, 2))

        if self.results is not None:
            self.results.append(variances)

        metadata.set_bitmap(bitmap)
        metadata.set_variances(variances[bitmap])
        metadata.set_all_variances(variances)

        # Print tensor shapes as a sanity check
        if self.do_print:
            print("Sent chunks :", top_chunks.shape)
            print("Sent indices:", top_chunks_indices)
            metadata.print_shapes()

        return (top_chunks, metadata)


class ChunkSelectGAbs(nn.Module):
    """Select which chunks will be transmitted, according to NN gradients

    Alternative method
    """

    def __init__(
        self,
        cr: float,
        g_mean: Tensor,
        results: List = None,
        do_print: bool = False,
    ):
        super().__init__()

        self.cr = cr
        self.g_mean = g_mean
        self.results = results
        self.chunk_split = ChunkSplit(None, do_print=do_print)
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        chunks = inp
        g_chunks = self.chunk_split(
            YuvImage(
                self.g_mean[0],
                self.g_mean[1],
                self.g_mean[2],
            ),
            metadata,
        )[0]

        gx = (g_chunks * chunks.abs()).sum(dim=(1, 2))

        # Figure out which chunks to send
        ntotal = chunks.shape[0]
        nsend = math.floor(ntotal * self.cr)

        if self.do_print:
            print("total chunks: {}, to send: {}".format(ntotal, nsend))

        _, top_chunks_indices = gx.topk(nsend)
        top_chunks_indices, _ = top_chunks_indices.sort()
        top_chunks = chunks[top_chunks_indices]

        bitmap = torch.zeros_like(gx, dtype=torch.bool)
        bitmap[top_chunks_indices] = True

        # Extract variance of each chunk
        variances = chunks.var(dim=(1, 2))

        if self.results is not None:
            self.results.append(variances)

        metadata.set_bitmap(bitmap)
        metadata.set_variances(variances[bitmap])
        metadata.set_all_variances(variances)

        # Print tensor shapes as a sanity check
        if self.do_print:
            print("Sent chunks :", top_chunks.shape)
            print("Sent indices:", top_chunks_indices)
            metadata.print_shapes()

        return (top_chunks, metadata)


class ChunkRestore(nn.Module):
    """Restore the chunks masked by a bitmap"""

    def __init__(self, device: torch.device, is_half: bool, do_print: bool = False):
        super().__init__()
        self.device = device
        self.is_half = is_half
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        nchunks = metadata.bitmap.shape[0]
        chunk_h, chunk_w = metadata.chunk_size

        # Restore sent chunks to the full size according to the bitmap
        chunks_full = torch.zeros(nchunks, chunk_h, chunk_w).to(
            self.device, non_blocking=True
        )
        if self.is_half:
            chunks_full = chunks_full.half()

        chunks_full.masked_scatter_(metadata.bitmap[:, None, None], inp)

        if self.do_print:
            print("Restored chunks: ", chunks_full.shape)

        return (chunks_full, metadata)


class ChunkSplit(nn.Module):
    """Split input samples into chunks"""

    def __init__(self, dct_size: Tuple[int, int] | None, do_print: bool = False):
        super().__init__()
        self.dct_size = dct_size
        self.do_print = do_print

    def forward(self, inp: YuvImage, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        if self.dct_size is not None:
            chunk_h, chunk_w = self.dct_size
        else:
            chunk_h, chunk_w = metadata.chunk_size

        _, img_h, img_w = inp.shape

        if ((img_h % chunk_h) != 0) or ((img_w % chunk_w) != 0):
            raise ValueError(
                "DCT: Image size {}x{} not divisible by chunk size {}x{}".format(
                    img_w, img_h, chunk_w, chunk_h
                )
            )

        # TODO: Weights according to channel importance
        wY, wU, wV = 1, 1, 1
        # TODO: Weights according to DCT coeff. importance for NN

        # Split input samples of each channel to chunks, scale them and concatenate
        # them along one dimension
        chunks_y = (
            inp.y.unfold(0, chunk_h, chunk_h)
            .unfold(1, chunk_w, chunk_w)
            .reshape((-1, chunk_h, chunk_w))
            * wY
        )
        chunks_u = (
            inp.u.unfold(0, chunk_h, chunk_h)
            .unfold(1, chunk_w, chunk_w)
            .reshape((-1, chunk_h, chunk_w))
            * wU
        )
        chunks_v = (
            inp.v.unfold(0, chunk_h, chunk_h)
            .unfold(1, chunk_w, chunk_w)
            .reshape((-1, chunk_h, chunk_w))
            * wV
        )
        chunks = torch.cat([chunks_y, chunks_u, chunks_v])

        if self.do_print:
            print(
                "split: inp shape: Y: {}, U: {}, V: {}".format(
                    inp.y.shape, inp.u.shape, inp.v.shape
                )
            )
            print(
                "split: out chunks:\n  Y: {}\n  U: {}\n  V: {}\n  total: {} ".format(
                    chunks_y.shape, chunks_u.shape, chunks_v.shape, chunks.shape
                )
            )

        return (chunks, metadata)


class ChunkCombine(nn.Module):
    """Combine chunks into a full image"""

    def __init__(
        self,
        mode: int,
        device: torch.device | str,
        dct_size: Tuple[int, int] | None,
        is_half: bool,
        do_print: bool = False,
    ):
        super().__init__()

        self.mode = mode
        self.device = device
        self.dct_size = dct_size
        self.is_half = is_half
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[YuvImage, Metadata]:
        padded_h, padded_w = metadata.padded_size

        if self.dct_size is not None:
            chunk_h, chunk_w = self.dct_size
        else:
            chunk_h, chunk_w = metadata.chunk_size

        ncw_y, nch_y = int(padded_w / chunk_w), int(padded_h / chunk_h)
        if self.mode == 420:
            ncw_u, nch_u = int(padded_w / 2 / chunk_w), int(padded_h / 2 / chunk_h)
            ncw_v, nch_v = int(padded_w / 2 / chunk_w), int(padded_h / 2 / chunk_h)
        else:
            ncw_u, nch_u = int(padded_w / chunk_w), int(padded_h / chunk_h)
            ncw_v, nch_v = int(padded_w / chunk_w), int(padded_h / chunk_h)

        nchunks_y = ncw_y * nch_y
        nchunks_u = ncw_u * nch_u
        nchunks_v = ncw_v * nch_v

        if self.do_print:
            print("Combine: inp size: ", inp.shape)
            print("Combine: dct size: ", self.dct_size)
            print("Combine: chunk size: ", chunk_h, chunk_w)
            print("Combine: nchunks: ", nchunks_y, nchunks_u, nchunks_v)

        chunks_y, chunks_u, chunks_v = inp.split((nchunks_y, nchunks_u, nchunks_v))

        if self.do_print:
            print(
                "Combine: chunks_y/u/v size: ",
                chunks_y.shape,
                chunks_u.shape,
                chunks_v.shape,
            )

        chunks_y_cols = chunks_y.reshape((nch_y, ncw_y, chunk_h, chunk_w)).unbind(1)
        y_cols = [torch.vstack(cy_col.unbind(0)) for cy_col in chunks_y_cols]

        chunks_u_cols = chunks_u.reshape((nch_u, ncw_u, chunk_h, chunk_w)).unbind(1)
        u_cols = [torch.vstack(cu_col.unbind(0)) for cu_col in chunks_u_cols]

        chunks_v_cols = chunks_v.reshape((nch_v, ncw_v, chunk_h, chunk_w)).unbind(1)
        v_cols = [torch.vstack(cv_col.unbind(0)) for cv_col in chunks_v_cols]

        return (
            YuvImage(torch.hstack(y_cols), torch.hstack(u_cols), torch.hstack(v_cols)),
            metadata,
        )


class PowerAllocate(nn.Module):
    """Scale chunks according to power allocation coefficient"""

    def __init__(self, power: float, results: List = None, do_print: bool = False):
        super().__init__()

        self.power = power
        self.results = results
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        sum_var = metadata.variances.sqrt().sum()

        beta = metadata.variances.pow(-1 / 4) * torch.sqrt(self.power / sum_var)
        beta[beta.isinf()] = 1.0

        if self.do_print:
            print("beta:", beta.shape)
        if self.results is not None:
            self.results.append(beta)

        return (torch.mul(inp, beta[:, None, None]), metadata)


class PowerAllocateG(nn.Module):
    """Scale chunks according to gradient-optimized power allocation coefficient

    Tries to minimize the loss distortion given max. transmission power.
    """

    def __init__(
        self, power: float, g_norm: Tensor, results: List = None, do_print: bool = False
    ):
        super().__init__()

        self.power = power
        self.g_norm = g_norm.reshape(-1)
        self.results = results
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        g_norm = self.g_norm[metadata.bitmap]

        sum_var = (metadata.variances.sqrt() * g_norm).sum()
        beta = metadata.variances.pow(-1 / 4) * torch.sqrt(
            self.power * g_norm / sum_var
        )
        beta[beta.isinf()] = 1.0

        if self.do_print:
            print("beta:", beta.shape)

        if self.results is not None:
            self.results.append(beta)

        return (torch.mul(inp, beta[:, None, None]), metadata)


class PowerAllocateGAbs(nn.Module):
    """Scale chunks according to gradient-optimized power allocation coefficient

    Tries to minimize the loss distortion given max. transmission power.
    """

    def __init__(
        self, power: float, g_norm: Tensor, results: List = None, do_print: bool = False
    ):
        super().__init__()

        self.power = power
        self.g_norm = g_norm.reshape(-1)
        self.results = results
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        g_norm = self.g_norm[metadata.bitmap]

        sum_var = (metadata.variances * g_norm.square()).pow(1 / 3).sum()
        beta = torch.sqrt(self.power / sum_var) * (g_norm / metadata.variances).pow(
            1 / 3
        )
        beta[beta.isinf()] = 1.0

        if self.do_print:
            print("beta:", beta.shape)

        if self.results is not None:
            self.results.append(beta)

        return (torch.mul(inp, beta[:, None, None]), metadata)


class PowerAllocateGMinPower(nn.Module):
    """Scale chunks according to gradient-optimized power allocation coefficient.

    Tries to minimize the transmission power given a max. loss distortion constraint.
    """

    def __init__(
        self, max_dist: float, g_norm: Tensor, snr_db: int, do_print: bool = False
    ):
        super().__init__()

        self.max_dist = max_dist
        self.g_norm = g_norm
        self.snr = math.pow(10, snr_db / 10)
        self.do_print = do_print

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        g_norm = self.g_norm
        g_norm_sent = g_norm[metadata.bitmap]
        g_norm_discarded = g_norm[~metadata.bitmap]
        variances_sent = metadata.variances
        variances_discarded = metadata.all_variances[~metadata.bitmap]

        # TODO: How to get sigma_noise?
        # inp_power = inp.square().mean()
        # noise_power = inp_power / self.snr
        # sigma_noise = noise_power.sqrt()

        sum_var_sent = (variances_sent.sqrt() * g_norm_sent).sum()
        sum_var_discarded = (variances_discarded.sqrt() * g_norm_discarded).sum()
        beta = (
            sigma_noise
            * (g_norm.square() / metadata.variances).pow(-1 / 4)
            * torch.sqrt(sum_var_sent / (self.max_dist - sum_var_discarded))
        )
        beta[beta.isinf()] = 1.0

        if self.do_print:
            print("beta:", beta.shape)

        return (torch.mul(inp, beta[:, None, None]), metadata)


class RandomOrthogonal(nn.Module):
    """Subtract mean from the input samples and store it in metadata"""

    def __init__(
        self, seed: int, device: torch.device, invert: bool, is_half: bool = False
    ):
        super().__init__()

        self.seed = seed
        self.device = device
        self.invert = invert
        self.is_half = is_half

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        torch.manual_seed(self.seed)

        nchunks = inp.shape[-3]

        H = torch.empty(nchunks, nchunks).to(self.device, non_blocking=True)
        nn.init.orthogonal_(H)

        if self.is_half:
            H = H.half()

        if self.invert:
            H = H.T

        return (torch.matmul(H, inp.reshape(nchunks, -1)).view(inp.shape), metadata)


class ZfEstimate(nn.Module):
    """Estimate the original values according to the ZF scheme"""

    def __init__(self, power: float):
        super().__init__()

        self.power = power

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        sum_var = metadata.variances.sqrt().sum()

        beta = metadata.variances.pow(-1 / 4) * torch.sqrt(self.power / sum_var)
        beta[beta.isinf()] = 1.0

        return (torch.div(inp, beta[:, None, None]), metadata)


class ZfEstimateG(nn.Module):
    """Estimate the original values according to the gradient-optimized ZF scheme"""

    def __init__(self, power: float, g_norm: Tensor):
        super().__init__()

        self.power = power
        self.g_norm = g_norm.reshape(-1)

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        g_norm = self.g_norm[metadata.bitmap]

        sum_var = (metadata.variances.sqrt() * g_norm).sum()
        beta = metadata.variances.pow(-1 / 4) * torch.sqrt(
            self.power * g_norm / sum_var
        )
        beta[beta.isinf()] = 1.0

        return (torch.div(inp, beta[:, None, None]), metadata)


class ZfEstimateGAbs(nn.Module):
    """Estimate the original values according to the gradient-optimized ZF scheme"""

    def __init__(self, power: float, g_norm: Tensor):
        super().__init__()

        self.power = power
        self.g_norm = g_norm.reshape(-1)

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        g_norm = self.g_norm[metadata.bitmap]

        sum_var = (metadata.variances * g_norm.square()).pow(1 / 3).sum()
        beta = torch.sqrt(self.power / sum_var) * (g_norm / metadata.variances).pow(
            1 / 3
        )
        beta[beta.isinf()] = 1.0

        return (torch.div(inp, beta[:, None, None]), metadata)


class LlseEstimate(nn.Module):
    """Estimate the original values according to the LLSE scheme"""

    def __init__(self, power: float):
        super().__init__()

        self.power = power

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        sum_var = metadata.variances.sqrt().sum()

        beta = metadata.variances.pow(-1 / 4) * torch.sqrt(self.power / sum_var)
        beta[beta.isinf()] = 1.0

        alpha = (
            beta
            * metadata.variances
            / (beta.square() * metadata.variances + metadata.noise_power)
        )
        alpha[alpha.isnan()] = 1.0

        return (inp * alpha[:, None, None], metadata)


class LlseEstimateG(nn.Module):
    """Estimate the original values according to the gradient-optimized LLSE scheme"""

    def __init__(self, power: float, g_norm: Tensor):
        super().__init__()

        self.power = power
        self.g_norm = g_norm.reshape(-1)

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        g_norm = self.g_norm[metadata.bitmap]

        sum_var = (metadata.variances.sqrt() * g_norm).sum()
        beta = metadata.variances.pow(-1 / 4) * torch.sqrt(
            self.power * g_norm / sum_var
        )
        beta[beta.isinf()] = 1.0

        tmp = beta * g_norm.square() * metadata.variances
        alpha = tmp / (beta * tmp + metadata.noise_power * g_norm.square())
        alpha[alpha.isnan()] = 1.0

        return (inp * alpha[:, None, None], metadata)


class LlseEstimateGAbs(nn.Module):
    """Estimate the original values according to the gradient-optimized LLSE scheme"""

    def __init__(self, power: float, g_norm: Tensor):
        super().__init__()

        self.power = power
        self.g_norm = g_norm.reshape(-1)

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        g_norm = self.g_norm[metadata.bitmap]

        sum_var = (metadata.variances * g_norm.square()).pow(1 / 3).sum()
        beta = torch.sqrt(self.power / sum_var) * (g_norm / metadata.variances).pow(
            1 / 3
        )
        beta[beta.isinf()] = 1.0

        tmp = beta * g_norm.square() * metadata.variances
        alpha = tmp / (beta * tmp + metadata.noise_power * g_norm.square())
        alpha[alpha.isnan()] = 1.0

        return (inp * alpha[:, None, None], metadata)


class LvcEncode(nn.Module):
    """Encode image using LVC"""

    def __init__(
        self,
        steps: List,
        references: dict,
        results: dict,
        chunk_size: Tuple[int, int] | int,
        do_print: bool = False,
    ):
        super().__init__()

        self.linears = nn.ModuleList(steps)
        self.references = references
        self.results = results
        self.chunk_size = chunk_size
        self.do_print = do_print

    def forward(self, inp) -> List[Tuple[Tensor, Metadata]]:
        res = []

        try:
            # assume input is a tensor
            if len(inp.shape) == 3:
                # reshape into 4D batch dimension
                nch, h, w = inp.shape
                inp = inp.reshape(-1, nch, h, w)
            elif len(inp.shape) != 4:
                raise ValueError(
                    "LvcEncoder: Unexpected input tensor shape: {}".format(inp.shape)
                )
        except AttributeError:
            # assume input is a list or some other iterable
            pass

        for i, image in enumerate(inp):
            if self.do_print:
                print("---- Encoding Image {}/{} ----".format(i + 1, len(inp)))
                print("enc inp shape: {}".format(inp[0].shape))
                print("enc inp min: {}, max: {}".format(inp[0].min(), inp[0].max()))

            try:
                image_size = (image.shape[-2], image.shape[-1])
            except TypeError:
                image_size = (image.shape()[-2], image.shape()[-1])

            chunk_size, nchunks = get_chunk_size(
                self.chunk_size, image_size[0], image_size[1]
            )
            if self.do_print:
                print(
                    "inp chunk size: {}x{}, nchunks: {}x{}".format(
                        chunk_size[1], chunk_size[0], nchunks[1], nchunks[0]
                    )
                )

            res_image = image
            res_metadata = Metadata(image_size, chunk_size, nchunks)

            for layer in self.linears:
                res_image, res_metadata = layer(res_image, res_metadata)

            res.append((res_image, res_metadata))

        self.metadatas = [metadata for _, metadata in res]

        return res

    def report(self, n: int = 0):
        # mean report
        try:
            ref = self.references["mean"][n]
            res = self.results["mean"][n]
            print("mean diff:  ", ref.sub(res))
        except (TypeError, KeyError) as err:
            print("mean diff:  ", "N/A ", "Error: ", err)

        # DCT report
        try:
            ref = self.references["dct"][n]
            res = self.results["dct"][n]
            print("DCT MAE:    ", ref.diff(res).abs().mean())
        except (TypeError, KeyError, IndexError) as err:
            print("DCT MAE:    ", "N/A ", "Error: ", err)

        # variance report
        try:
            ref = self.references["var"][n]
            res = self.results["var"][n]
            print("var MAE:    ", ref.sub(res).abs().mean())
        except (TypeError, KeyError, RuntimeError, IndexError) as err:
            print("var MAE:  ", "N/A ", "Error: ", err)

        # power allocation report
        try:
            ref = self.references["g"][n]
            res = self.results["g"][n].sort().values
            print("g MAE:      ", ref.sub(res).abs().mean())
        except (TypeError, KeyError, RuntimeError, IndexError) as err:
            print("g MAE:    ", "N/A ", "Error: ", err)


class Channel(nn.Module):
    """Simulate the channel by adding AWGN to the channel input"""

    def __init__(
        self,
        snr_db: int,
        device: torch.device,
        is_half: bool,
        packet_loss=None,
        seed: int = None,
        do_print: bool = False,
        results: List = None,
    ):
        super().__init__()

        self.device = device
        self.is_half = is_half
        self.packet_loss = packet_loss
        self.seed = seed
        self.do_print = do_print
        self.results = results
        self.snr = math.pow(10, snr_db / 10)

        if self.do_print:
            print("Channel SNR: {} dB ({})".format(snr_db, self.snr))

    def forward(
        self, inp: List[Tuple[Tensor, Metadata]]
    ) -> List[Tuple[Tensor, Metadata]]:
        channel_out = []

        for i, (chunks, metadata) in enumerate(inp):
            if self.do_print:
                print("---- Channel Image {}/{} ----".format(i + 1, len(inp)))

            inp_power = chunks.square().mean()

            noise_power = inp_power / self.snr
            sigma_noise = noise_power.sqrt()

            metadata.set_noise_power(noise_power)

            noise = torch.randn(chunks.shape).to(self.device, non_blocking=True)

            if self.is_half:
                noise = noise.half()

            chunks_out = chunks + sigma_noise * noise

            if self.results is not None:
                self.results.append(sigma_noise * noise)

            zero_idxs = None
            if self.packet_loss == "dc":
                zero_idxs = torch.tensor(0)
            elif self.packet_loss is not None:
                if self.seed is not None:
                    torch.manual_seed(self.seed)

                nchunks = chunks.shape[-3]
                num_dropped_chunks = round(nchunks * self.packet_loss)
                zero_idxs = (
                    torch.randperm(nchunks)[:num_dropped_chunks]
                    .to(self.device, non_blocking=True)
                    .sort()
                    .values
                )

            if zero_idxs is not None:
                chunks_out.index_fill_(0, zero_idxs, 0.0)

                if self.do_print:
                    print("Zero chunks :", zero_idxs)

            channel_out.append((chunks_out, metadata))

            if self.do_print:
                print("Data power  :", inp_power)
                print("Noise power :", noise_power)

        return channel_out


class SionnaConfig(TypedDict):
    nbits_per_sym: int  # no. bits per symbol, 2 = 4-QAM, 4 = 16-QAM, etc.
    coderate: float


def fmt_sionna(config: SionnaConfig) -> str:
    return f"{2**config['nbits_per_sym']:3}-QAM|{config['coderate']:4.2}"


class SionnaChannel(nn.Module):
    """Simulate the channel by adding AWGN to the channel input"""

    def __init__(
        self,
        snr_db: int,
        device: torch.device | str,
        seed: int,
        config: SionnaConfig,
        digital: bool = False,
        do_print: bool = False,
        dry: bool = False,
        num_threads: int | None = None,
    ):
        super().__init__()

        self.snr_db = snr_db
        self.device = device
        self.seed = seed
        self.digital = digital
        self.do_print = do_print
        self.dry = dry
        self.num_threads = num_threads
        self.snr = math.pow(10, snr_db / 10)

        if self.do_print:
            print("Initializing Sionna channel")
            print("-- Channel SNR: {} dB ({})".format(snr_db, self.snr))

        if not self.digital:
            if config["nbits_per_sym"] != 2:
                raise ValueError(
                    f"Number of bits per symbol must be 2 with analog "
                    f"channel. Got {config['nbits_per_sym']}"
                )

            if config["coderate"] != 1.0:
                raise ValueError(
                    f"Coderate must be 1.0 with analog channel. Got {config['coderate']}"
                )

        self.nbits_per_sym = config["nbits_per_sym"]
        self.coderate = config["coderate"]

        if self.do_print:
            print(
                f"-- Digital: {self.digital}, coderate: {self.coderate}, nbits_per_sym: {self.nbits_per_sym}"
            )

    def forward(
        self, inp: List[Tuple[Tensor, Metadata]]
    ) -> List[Tuple[Tensor, Metadata]]:
        channel_out = []

        # Re-creating the channel each time for reproducibility
        import tensorflow as tf
        from digcom import OFDMSystemRaw

        tf.get_logger().setLevel("ERROR")
        tf.random.set_seed(self.seed)

        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = f"{self.num_threads}"
            tf.config.threading.set_intra_op_parallelism_threads(self.num_threads)
            tf.config.threading.set_inter_op_parallelism_threads(self.num_threads)

        self.tf = tf
        self.ofdm_system_raw = OFDMSystemRaw

        for i, (symbols, metadata) in enumerate(inp):
            symbols_out = self._run_image(symbols)
            channel_out.append((symbols_out, metadata))

        return channel_out

    def _run_image(self, symbols: Tensor) -> Tensor:
        # Set some configuration constants
        self.fft_size = 192
        num_ofdm_symbols = 14
        pilot_ofdm_symbol_indices = [2, 11]

        resource_grid_config = {
            "num_ofdm_symbols": num_ofdm_symbols,
            "fft_size": self.fft_size,
            "pilot_ofdm_symbol_indices": pilot_ofdm_symbol_indices,
        }

        self.num_ut = 1  # number of user terminals (UT)
        self.num_ut_ant = 1  # number of UT antennas
        self.num_nonpilot_ofdm_symbols = num_ofdm_symbols - len(
            pilot_ofdm_symbol_indices
        )

        # Set batch size based on the number of symbols
        codeword_size = self.fft_size * self.num_nonpilot_ofdm_symbols
        self.num_info_bits = self.coderate * codeword_size * self.nbits_per_sym

        self.num_symbols = np.prod(symbols.shape)
        self.BATCH_SIZE = math.ceil(self.num_symbols / self.num_info_bits)

        # Create the tranmission model
        self.model = self.ofdm_system_raw(
            self.BATCH_SIZE,
            resource_grid_config,
            self.num_ut,
            self.num_ut_ant,
            digital=self.digital,
            perfect_csi=False,
            coderate=self.coderate,
            nbits_per_sym=self.nbits_per_sym,
            show=False,
        )

        if self.digital:
            symbols_out = self._run_img_digital(symbols)
        else:
            symbols_out = self._run_img_analog(symbols)

        return symbols_out

    def _run_img_digital(self, symbols: Tensor) -> Tensor:
        self.num_padded = math.ceil(self.BATCH_SIZE * self.num_info_bits)

        if self.num_padded % 2 != 0:
            self.num_padded += 1

        padded = self.tf.pad(
            np.reshape(symbols.cpu().numpy(), -1),
            [[0, self.num_padded - self.num_symbols]],
        )

        inp = self.tf.reshape(
            padded,
            [
                self.BATCH_SIZE,
                self.num_ut,
                self.num_ut_ant,
                int(self.num_padded / self.BATCH_SIZE),
            ],
        )

        if self.dry:
            out = self.tf.zeros_like(inp)
        else:
            EBN0_DB = self.snr_db  # TODO: Calculcate correctly
            out = self.model.run_with_input(inp, EBN0_DB)

        symbols_out = torch.tensor(out.numpy()).to(self.device, non_blocking=True)
        symbols_out = symbols_out.reshape(-1)[: self.num_symbols]

        return symbols_out

    def _run_img_analog(self, symbols: Tensor) -> Tensor:
        # Scale coefficients to have variance == 1.0
        symbols_std = symbols.std()
        symbols_norm = symbols / symbols_std
        symbols_norm = self.tf.constant(symbols_norm.cpu().numpy())

        # Pad data to the required size and flatten
        self.num_padded = (
            self.BATCH_SIZE * 2 * self.fft_size * self.num_nonpilot_ofdm_symbols
        )

        if self.num_padded % 2 != 0:
            self.num_padded += 1

        padded = self.tf.pad(
            np.reshape(symbols_norm, -1), [[0, self.num_padded - self.num_symbols]]
        )

        # Assign pair of samples as real/image values of a complex tensor
        inp_re = padded[::2]
        inp_im = padded[1::2]
        inp = self.tf.complex(inp_re, inp_im)

        # Reshape for Sionna processing (assuming 1 TX device with 1 antenna and
        # 1 stream per antenna)
        inp = self.tf.reshape(
            inp,
            [
                self.BATCH_SIZE,
                self.num_ut,
                self.num_ut_ant,
                self.fft_size * self.num_nonpilot_ofdm_symbols,
            ],
        )

        # Run simulation
        if self.dry:
            out = self.tf.zeros_like(inp)
        else:
            EBN0_DB = self.snr_db  # TODO: Calculcate correctly
            out = self.model.run_with_input(inp, EBN0_DB)

        # Flatten the output and extract real/imag values
        out = self.tf.reshape(out, -1)
        out_re = self.tf.math.real(out)
        out_im = self.tf.math.imag(out)

        # Reconstruct the output image from the real/imag values
        symbols_out_norm = np.zeros_like(padded)
        symbols_out_norm[::2] = out_re.numpy()
        symbols_out_norm[1::2] = out_im.numpy()
        symbols_out_norm = symbols_out_norm[: self.num_symbols].reshape(symbols.shape)

        # Scale data back to original range
        symbols_out = symbols_out_norm * symbols_std.item()
        symbols_out = torch.tensor(symbols_out).to(self.device, non_blocking=True)

        return symbols_out


class NoopChannel(nn.Module):
    """Just passes the input, used as placeholder"""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self, inp: List[Tuple[Tensor, Metadata]]
    ) -> List[Tuple[Tensor, Metadata]]:
        return inp


class LvcDecode(nn.Module):
    """Decode LVC stream"""

    def __init__(self, steps: List):
        super().__init__()

        self.linears = nn.ModuleList(steps)

    def forward(self, inp: List[Tuple[Tensor, Metadata]]) -> List:
        res_images = []

        for image, metadata in inp:
            res = image

            for layer in self.linears:
                res, metadata = layer(res, metadata)

            res_images.append(res)

        return res_images


class Stack(nn.Module):
    """Stack a list of images to one tensor"""

    def __init__(self, unsqueeze: bool = False):
        super().__init__()
        self.unsqueeze = unsqueeze

    def forward(self, inp: List[Tensor]) -> Tensor:
        res = torch.stack([img for img in inp])

        if self.unsqueeze:
            res = res.reshape(3, inp[0].shape[-2], inp[0].shape[-1])

        return res


class Unstack(nn.Module):
    """Unstack a tensor into a list of images"""

    def __init__(self, unsqueeze: bool = False):
        super().__init__()
        self.unsqueeze = unsqueeze

    def forward(self, inp: Tensor) -> List[Tensor]:
        if self.unsqueeze:
            return [inp]
        else:
            return list(inp.unbind())


class SaveMetadata(nn.Module):
    """Saves metadata to internal state and output only tensors"""

    def __init__(self, metadatas: List[Metadata]):
        super().__init__()
        self.metadatas = metadatas

    def forward(self, inp: List[Tuple[Tensor, Metadata]]) -> List[Tensor]:
        self.metadatas = [t[1] for t in inp]
        return [t[0] for t in inp]


class LoadMetadata(nn.Module):
    """Create metadata and attach it to the input tensor"""

    def __init__(self, metadatas: List[Metadata]):
        super().__init__()
        self.metadatas = metadatas

    def forward(self, inp: List[Tensor]) -> List[Tuple[Tensor, Metadata]]:
        return list(zip(inp, self.metadatas))


class EncodeTurboJPEG(nn.Module):
    """Encode RGB image as JPEG using TurboJPEG"""

    def __init__(
        self,
        mode: int,
        quality: int,
        device: str | torch.device = "cpu",
        jpeg_lib: str | Path = DEFAULT_JPEG_LIB,
        results: List | None = None,
    ):
        super().__init__()

        if mode != 420:
            raise ValueError(f"Encode TurboJPEG: Mode {mode} not supported for now")

        self.mode = mode
        self.quality = quality
        self.device = device
        self.jpeg = TurboJPEG(jpeg_lib)
        self.results = results

    def forward(self, inp_rgb: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        # assume (N)CHW fomat
        (_, img_h, img_w) = inp_rgb.shape

        # CHW -> HWC expected by OpenCV
        inp_rgb = inp_rgb * 255.0
        inp_rgb = inp_rgb.permute(1, 2, 0).type(torch.uint8)
        inp_rgb = inp_rgb.detach().cpu().numpy()

        inp_yuv420 = cv.cvtColor(inp_rgb, cv.COLOR_RGB2YUV_YV12)
        jpeg_bytes = self.jpeg.encode_from_yuv(
            inp_yuv420, img_h, img_w, self.quality, TJSAMP_420
        )

        jpeg_bits = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8)).astype(
            np.float32
        )
        out = torch.tensor(jpeg_bits).to(self.device)

        metadata.img_w = img_w
        metadata.img_h = img_h

        if self.results is not None:
            self.results.append(
                {
                    "img_w": metadata.img_w,
                    "img_h": metadata.img_h,
                    "enc_size": len(jpeg_bytes),
                }
            )

        return (out, metadata)


class DecodeTurboJPEG(nn.Module):
    """Decode JPEG bitstream into RGB using TurboJPEG"""

    def __init__(
        self,
        mode: int,
        device: str | torch.device = "cpu",
        jpeg_lib: str | Path = DEFAULT_JPEG_LIB,
    ):
        super().__init__()

        if mode != 420:
            raise ValueError(f"Decode TurboJPEG: Mode {mode} not supported for now")

        self.mode = mode
        self.device = device
        self.jpeg = TurboJPEG(jpeg_lib)

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        jpeg_bits = inp.detach().cpu().numpy()
        jpeg_bytes = np.packbits(jpeg_bits.astype(np.uint8))

        try:
            img_yuv420, _ = self.jpeg.decode_to_yuv(jpeg_bytes)
        except OSError:
            # image failed to decode: Replace with zeros
            img_yuv420 = np.zeros(
                int(metadata.img_w * metadata.img_h * 3 / 2), dtype=np.uint8
            )

        img_rgb = cv.cvtColor(
            img_yuv420.reshape(int(metadata.img_h * 3 / 2), metadata.img_w),
            cv.COLOR_YUV2RGB_YV12,
        )
        img_rgb = torch.tensor(img_rgb, dtype=torch.float).to(self.device) / 255.0

        # HWC -> CHW
        img_rgb = img_rgb.permute(2, 0, 1)

        return (img_rgb, metadata)


class EncodeJPEG(nn.Module):
    """Encode RGB image as JPEG using Torchjpeg"""

    def __init__(
        self,
        mode: int,
        quality: int,
        device: str | torch.device = "cpu",
        results: List | None = None,
    ):
        super().__init__()

        if mode != 444:
            raise ValueError(f"Encode JPEG: Mode {mode} not supported for now")

        self.mode = mode
        self.quality = quality
        self.device = device
        self.results = results
        self.color_samp_factor_vertical = 1
        self.color_samp_factor_horizontal = 1

    def forward(self, inp_rgb: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        _, _, _, _, jpeg_bytes = torchjpeg.codec.quantize_at_quality(
            inp_rgb,
            self.quality,
            self.color_samp_factor_vertical,
            self.color_samp_factor_horizontal,
        )

        # jpeg_bits = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8)).astype(
        jpeg_bits = np.unpackbits(jpeg_bytes.detach().cpu().numpy()).astype(np.float32)
        out = torch.tensor(jpeg_bits).to(self.device)

        metadata.img_w = inp_rgb.shape[-1]
        metadata.img_h = inp_rgb.shape[-2]

        if self.results is not None:
            self.results.append(
                {
                    "img_w": metadata.img_w,
                    "img_h": metadata.img_h,
                    "enc_size": jpeg_bytes.numel(),
                }
            )

        return (out, metadata)


class EncodeGRACE(nn.Module):
    """Encode a RGB image as GRACE-modified JPEG using Torchjpeg"""

    def __init__(
        self,
        mode: int,
        B: float,
        g_block_norm: Tensor,
        w: Tensor,
        device: str | torch.device = "cpu",
        results: List | None = None,
    ):
        super().__init__()

        if mode != 444:
            raise ValueError(f"Encode GRACE: Mode {mode} not supported for now")

        if g_block_norm.shape != torch.Size([3, 8, 8]):
            raise ValueError(f"g_norm shape must be (3,8,8), got {g_block_norm.shape}")

        self.mode = mode
        self.B = B
        self.g_block_norm = g_block_norm
        self.device = device
        self.w = w.to(self.device)
        self.results = results
        self.color_samp_factor_vertical = 1
        self.color_samp_factor_horizontal = 1
        self.rgb2yuv = RgbToYcbcr(self.w)

        self.quant_tables = (
            torch.zeros_like(g_block_norm)
            .type(torch.int16)
            .to(self.device, non_blocking=True)
        )

        for i, g_norm in enumerate(self.g_block_norm):
            qt, _ = get_quant_table_approx(g_norm, self.B)
            self.quant_tables[i] = qt.to(self.device, non_blocking=True)

        self.quant_tables = self.quant_tables.type(torch.int16)

    def forward(self, inp_rgb: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        inp_yuv = self.rgb2yuv(inp_rgb.to(self.device)).cpu()

        _, _, _, _, jpeg_bytes = torchjpeg.codec.quantize_at_quality_custom(
            inp_yuv.cpu(),
            100,
            self.quant_tables.cpu(),
            self.color_samp_factor_vertical,
            self.color_samp_factor_horizontal,
        )

        jpeg_bits = np.unpackbits(jpeg_bytes.detach().cpu().numpy()).astype(np.float32)
        out = torch.tensor(jpeg_bits).to(self.device)

        metadata.img_w = inp_rgb.shape[-1]
        metadata.img_h = inp_rgb.shape[-2]

        if self.results is not None:
            self.results.append(
                {
                    "img_w": metadata.img_w,
                    "img_h": metadata.img_h,
                    "enc_size": jpeg_bytes.numel(),
                }
            )

        return (out, metadata)


class DecodeJPEGGRACE(nn.Module):
    """Decode JPEG or GRACE bitstream into RGB using Torchjpeg"""

    def __init__(
        self,
        mode: int,
        w: Tensor,
        codec: Literal["jpeg", "grace"],
        device: str | torch.device = "cpu",
        do_print: bool = False,
    ):
        super().__init__()

        if mode != 444:
            raise ValueError(f"Decode JPEG: Mode {mode} not supported for now")

        self.mode = mode
        self.codec = codec
        self.device = device
        self.do_print = do_print
        self.w = w.to(self.device)
        self.yuv2rgb = YcbcrToRgb(self.w)

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        jpeg_bits = inp.detach().cpu().numpy()
        jpeg_bytes = np.packbits(jpeg_bits.astype(np.uint8))
        jpeg_bytes = torch.tensor(jpeg_bytes, dtype=torch.uint8)

        try:
            dimensions, quant_tables, y_coeffs, cbcr_coeffs = (
                torchjpeg.codec.read_coefficients_from_mem(jpeg_bytes)
            )
            error = False
        except:
            error = True

        if self.codec == "jpeg":
            raw = False
        elif self.codec == "grace":
            raw = True

        if error:
            if self.do_print:
                print(f"Error decoding JPEG/GRACE. Codec: {self.codec}")

            spatial_rgb = torch.zeros(3, metadata.img_h, metadata.img_w).to(self.device)
        else:
            spatial = torchjpeg.codec.reconstruct_full_image(
                y_coeffs, quant_tables, cbcr_coeffs, dimensions, raw=raw
            )
            spatial = spatial.to(self.device)

            if self.codec == "jpeg":
                spatial_rgb = spatial
            elif self.codec == "grace":
                spatial_rgb = self.yuv2rgb(spatial).clamp(0.0, 1.0)

        return (spatial_rgb, metadata)


class EncodeIcmTcm(nn.Module):
    """Encode RGB image using ICM"""

    def __init__(
        self,
        mode: int,
        model: ICM | TCM,
        results: List | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        if mode != 444:
            raise ValueError(f"Encode ICM/TCM: Mode {mode} not supported for now")

        self.mode = mode
        self.device = device
        self.p = 128
        self.icm = model
        self.results = results

    def forward(self, inp_rgb: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        x_padded, padding = icm_pad(inp_rgb.unsqueeze(0).to(self.device), self.p)
        out_enc = self.icm.compress(x_padded)

        sizes = []
        out_bytes = np.array([], dtype=np.uint8)
        for s in out_enc["strings"]:
            sz = []
            for ss in s:
                sz.append(len(ss))
                out_bytes = np.concatenate(
                    [out_bytes, np.frombuffer(ss, dtype=np.uint8)]
                )
            sizes.append(sz)

        out_bits = np.unpackbits(out_bytes).astype(np.float32)
        out = torch.tensor(out_bits).to(self.device)

        metadata.icm_shape = out_enc["shape"]
        metadata.icm_sizes = sizes
        metadata.icm_padding = padding
        metadata.img_w = inp_rgb.shape[-1]
        metadata.img_h = inp_rgb.shape[-2]

        if self.results is not None:
            self.results.append(
                {
                    "img_w": metadata.img_w,
                    "img_h": metadata.img_h,
                    "enc_size": torch.tensor(out_bytes).numel(),
                }
            )

        return (out, metadata)


class DecodeIcmTcm(nn.Module):
    """Decode JPEG or GRACE bitstream into RGB using Torchjpeg"""

    def __init__(
        self,
        mode: int,
        model: ICM | TCM,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        if mode != 444:
            raise ValueError(f"Decode JPEG: Mode {mode} not supported for now")

        self.mode = mode
        self.device = device
        self.p = 128
        self.icm = model

    def forward(self, inp: Tensor, metadata: Metadata) -> Tuple[Tensor, Metadata]:
        out_bits = inp.detach().cpu().numpy()
        out_bytes = np.packbits(out_bits.astype(np.uint8))
        out_bytes = torch.tensor(out_bytes, dtype=torch.uint8)

        sizes = metadata.icm_sizes
        strings = []
        prev_idx = 0

        for subsizes in sizes:
            substrings = []
            for subsize in subsizes:
                next_idx = prev_idx + subsize
                substrings.append(
                    bytes(out_bytes[prev_idx:next_idx].detach().cpu().numpy())
                )
                prev_idx = next_idx
            strings.append(substrings)

        out_dec = self.icm.decompress(strings, metadata.icm_shape)
        out = icm_crop(out_dec["x_hat"], metadata.icm_padding).squeeze()

        return (out, metadata)


def allowed_image_sizes(
    chunk_size: Tuple[int, int], mode: int, cr=1, min_dim=0, max_dim=None
):
    """Print out valid image sizes (according to MATLAB) given a chunk size and subsampling"""

    if mode != 444 and mode != 420:
        raise ValueError("Invalid subsampling mode {} (choose 444 or 420)".format(mode))

    h, w = chunk_size

    if not max_dim:
        max_dim = max(w, h) * 100

    min_W = math.floor(min_dim / w) * w
    min_H = math.floor(min_dim / h) * h

    W = max(w, min_W)
    H = max(h, min_H)

    while W <= max_dim:
        while H <= max_dim:
            num_chunks_y = (W / w) * (H / h)

            if mode == 420:
                num_chunks_u = (W / 2 / w) * (H / 2 / h)
                num_chunks_v = (W / 2 / w) * (H / 2 / h)
            else:
                num_chunks_u = (W / w) * (H / h)
                num_chunks_v = (W / w) * (H / h)

            num_chunks = cr * (num_chunks_y + num_chunks_u + num_chunks_v)

            if (
                num_chunks == 1
                or num_chunks == 2
                or (
                    num_chunks.is_integer()
                    and num_chunks > 2
                    and (num_chunks / 4).is_integer()
                    and (
                        math.log2(num_chunks).is_integer()
                        or math.log2(num_chunks / 12).is_integer()
                        or math.log2(num_chunks / 20).is_integer()
                    )
                )
            ):
                print(
                    "W: {:4.0f}, H: {:4.0f}, num_chunks: {:5.0f}".format(
                        W, H, num_chunks
                    )
                )

            H += h

        H = max(h, min_H)
        W += w


class GradConfig(TypedDict):
    """Encodes what tasks should be gradient-optimized and stores probed data"""

    type: Literal[
        "dist_abs",  # use distortion defined as abs. value of loss error
        "dist_precise",  # use distortion defined as squared of loss error (more precise)
        "dist_sq",  # use distortion defined as squared loss error
    ]
    g_yuv: bool
    g_select: bool
    g_allocate: bool
    w: torch.Tensor
    grad_mean: torch.Tensor
    grad_norm: torch.Tensor


class JPEGConfig(TypedDict):
    """Parameters for JPEG encoding"""

    codec: Literal["jpeg", "grace", "icm", "tcm"]
    param: int | float | str
    param_name: Literal["Q", "B", "model"]  # for JPEG, GRACE, and icm/tcm
    param_fmt: str  # how to format the found parameter
    turbojpeg_enc: bool
    turbojpeg_dec: bool


def create_lvc(
    lvc_params: dict,
    device: torch.device,
    half: bool,
    unsqueeze: bool = False,
    do_print: bool = False,
    results: dict = {"noise": None, "jpeg": None},
    grad_config: GradConfig | None = None,
    sionna_config: SionnaConfig | None = None,
    jpeg_config: JPEGConfig | None = None,
) -> transforms.Compose:
    norm = "ortho"
    power = 1.0
    w_human = torch.Tensor([0.299, 0.587, 0.114])

    yuv_mode = lvc_params["mode"]

    if grad_config is not None:
        for key in ["grad_mean", "grad_norm"]:
            grad = grad_config[key]

            if grad is not None:
                # Flatten (and subsample) gradients
                if yuv_mode == 420:
                    chunk_w, chunk_h = grad.shape[1:]
                    grad_y = grad[0, :, :].reshape(-1)
                    grad_u = grad[1, : chunk_h // 2, : chunk_w // 2].reshape(-1)
                    grad_v = grad[2, : chunk_h // 2, : chunk_w // 2].reshape(-1)
                    grad_config[key] = torch.cat([grad_y, grad_u, grad_v])
                else:
                    grad_config[key] = grad

        if grad_config["g_yuv"]:
            w = grad_config["w"]
        else:
            w = w_human

        grad_type = grad_config["type"]
        g_allocate = grad_config["g_allocate"]
        g_select = grad_config["g_select"]
    else:
        w = w_human
        grad_type = None
        g_allocate = False
        g_select = False

    references = None

    csnr_db = lvc_params["csnr_db"]
    cr = lvc_params["cr"]
    seed = lvc_params["seed"]
    packet_loss = lvc_params["packet_loss"]

    if lvc_params["estimator"] == "zf":
        if g_allocate and grad_config is not None:
            grad_norm = grad_config["grad_norm"].to(device, non_blocking=True)
            if grad_type == "dist_abs":
                estimator = ZfEstimateGAbs(power, grad_norm)
            else:
                estimator = ZfEstimateG(power, grad_norm)
        else:
            estimator = ZfEstimate(power)
    elif lvc_params["estimator"] == "llse":
        if g_allocate and grad_config is not None:
            grad_norm = grad_config["grad_norm"].to(device, non_blocking=True)
            if grad_type == "dist_abs":
                estimator = LlseEstimateGAbs(power, grad_norm)
            else:
                estimator = LlseEstimateG(power, grad_norm)
        else:
            estimator = LlseEstimate(power)
    else:
        raise ValueError("Unknown estimator. Use 'zf' or 'llse'.")

    try:
        dct_w = lvc_params["dct_w"]
        dct_h = lvc_params["dct_h"]
        if (dct_w is None) or (dct_h is None):
            dct_size = None
        else:
            if int(dct_w) != dct_w or int(dct_h) != dct_h:
                raise ValueError(f"Invalid floating point DCT size {dct_w}x{dct_h}")
            else:
                dct_size = (int(dct_h), int(dct_w))
    except KeyError:
        dct_size = None

    if dct_size is None:
        dct_layers = (
            DctYuvMetadata(half, norm),
            ChunkSplit(dct_size, do_print=do_print),
        )
        idct_layers = (
            ChunkCombine(yuv_mode, device, dct_size, is_half=half, do_print=do_print),
            IdctYuvMetadata(half, norm),
        )
    else:
        grouping = lvc_params["grouping"]
        dct_layers = (
            ChunkSplit(dct_size, do_print=do_print),
            DctBlock(
                dct_size, yuv_mode, grouping, is_half=half, norm=norm, do_print=do_print
            ),
        )
        idct_layers = (
            IdctBlock(
                dct_size, yuv_mode, grouping, is_half=half, norm=norm, do_print=do_print
            ),
            ChunkCombine(yuv_mode, device, dct_size, is_half=half, do_print=do_print),
        )

    try:
        chunk_size = lvc_params["nchunks"]
    except KeyError:
        chunk_w = lvc_params["chunk_w"]
        chunk_h = lvc_params["chunk_h"]
        chunk_size = (chunk_h, chunk_w)

    try:
        color_space = lvc_params["color_space"]
    except KeyError:
        color_space = "yuv"

    if color_space == "yuv":
        transform_fwd = RgbToYcbcrMetadata(w=w)
        transform_rev = YcbcrToRgbMetadata(w=w)
    elif color_space == "rgb":
        if yuv_mode != 444:
            raise ValueError(f"Mode {yuv_mode} is not allowed in RGB. Only 444.")
        transform_fwd = TensorToImage()
        transform_rev = ImageToTensor()
    else:
        raise ValueError(f"Unknown color space {color_space}.")

    if g_select and grad_config is not None:
        if grad_type == "dist_abs":
            grad_mean = grad_config["grad_mean"].to(device, non_blocking=True)
            chunk_select = ChunkSelectGAbs(cr, grad_mean, do_print=do_print)
        elif grad_type == "dist_precise":
            grad_mean = grad_config["grad_mean"].to(device, non_blocking=True)
            chunk_select = ChunkSelectGPrecise(cr, grad_mean, do_print=do_print)
        else:
            grad_norm = grad_config["grad_norm"].to(device, non_blocking=True)
            chunk_select = ChunkSelectG(cr, grad_norm, do_print=do_print)
    else:
        chunk_select = ChunkSelect(cr, do_print=do_print)

    if g_allocate and grad_config is not None:
        grad_norm = grad_config["grad_norm"].to(device, non_blocking=True)
        if grad_type == "dist_abs":
            power_alloc = PowerAllocateGAbs(power, grad_norm, do_print=do_print)
        else:
            power_alloc = PowerAllocateG(power, grad_norm, do_print=do_print)
    else:
        power_alloc = PowerAllocate(power, do_print=do_print)

    if jpeg_config is not None:
        if jpeg_config["codec"] == "grace":
            if grad_config is None:
                raise ValueError("GRACE encoder needs grad_norm")

            if jpeg_config["turbojpeg_enc"]:
                raise ValueError("TurboJPEG encoding not supported with GRACE")

            grad_norm = grad_config["grad_norm"].to(device, non_blocking=True)
            encoder_layers = [
                EncodeGRACE(
                    yuv_mode,
                    jpeg_config["param"],
                    grad_norm,
                    w,
                    device=device,
                    results=results["jpeg"],
                )
            ]
        elif jpeg_config["codec"] == "jpeg":
            if jpeg_config["turbojpeg_enc"]:
                encoder_layers = [
                    EncodeTurboJPEG(
                        yuv_mode,
                        jpeg_config["param"],
                        device=device,
                        results=results["jpeg"],
                    )
                ]
            else:
                encoder_layers = [
                    EncodeJPEG(
                        yuv_mode,
                        jpeg_config["param"],
                        device=device,
                        results=results["jpeg"],
                    )
                ]
        elif jpeg_config["codec"] == "icm" or jpeg_config["codec"] == "tcm":
            checkpoint = jpeg_config["param"]

            if jpeg_config["codec"] == "icm":
                model = ICM()
            elif jpeg_config["codec"] == "tcm":
                if "mse_lambda" in checkpoint:
                    N = 128
                else:
                    N = 64
                model = TCM(N=N)

            model.to(device)
            model.eval()
            checkpoint = torch.load(checkpoint, map_location=device)
            directory = {}
            for k, v in checkpoint["state_dict"].items():
                directory[k.replace("module.", "")] = v
            model.load_state_dict(directory)
            model.update()

            encoder_layers = [
                EncodeIcmTcm(yuv_mode, model, results=results["jpeg"], device=device)
            ]
            decoder_layers = [DecodeIcmTcm(yuv_mode, model, device=device)]

        if jpeg_config["codec"] in ("jpeg", "grace"):
            if jpeg_config["turbojpeg_dec"]:
                decoder_layers = [DecodeTurboJPEG(yuv_mode, device=device)]
            else:
                decoder_layers = [
                    DecodeJPEGGRACE(
                        yuv_mode,
                        w,
                        jpeg_config["codec"],
                        device=device,
                        do_print=do_print,
                    )
                ]

    else:
        encoder_layers = [
            ToDevice(device),
            Pad(yuv_mode, dct_size, do_print=do_print),
            transform_fwd,
            Downsample(mode=yuv_mode),
            SubtractMean(device),
            dct_layers[0],
            dct_layers[1],
            chunk_select,
            power_alloc,
            RandomOrthogonal(seed, device, invert=False, is_half=half),
        ]
        decoder_layers = [
            RandomOrthogonal(seed, device, invert=True, is_half=half),
            estimator,
            ChunkRestore(device=device, is_half=half, do_print=do_print),
            idct_layers[0],
            idct_layers[1],
            RestoreMean(),
            Upsample(mode=yuv_mode),
            transform_rev,
            Crop(do_print=do_print),
        ]

    lvc_encoder = LvcEncode(encoder_layers, references, None, chunk_size).to(
        device, non_blocking=True
    )
    lvc_decoder = LvcDecode(decoder_layers).to(device, non_blocking=True)

    stack = Stack(unsqueeze=unsqueeze).to(device, non_blocking=True)

    if half:
        lvc_encoder.half()
        lvc_decoder.half()
        stack.half()

    if csnr_db == "inf":
        lvc_chain = transforms.Compose([lvc_encoder, lvc_decoder, stack])
    elif sionna_config is not None:
        if jpeg_config is None:
            digital = False
        else:
            digital = True

        channel = SionnaChannel(
            csnr_db,
            device,
            seed,
            sionna_config,
            digital=digital,
            do_print=do_print,
        ).to(device, non_blocking=True)
        lvc_chain = transforms.Compose([lvc_encoder, channel, lvc_decoder, stack])
    else:
        channel = Channel(
            csnr_db,
            device,
            half,
            packet_loss=packet_loss,
            do_print=do_print,
            results=results["noise"],
        ).to(device, non_blocking=True)

        if half:
            channel.half()

        lvc_chain = transforms.Compose([lvc_encoder, channel, lvc_decoder, stack])

    return lvc_chain


def create_dct(
    dct_size: tuple,
    chunk_size: int,
    w_yuv: torch.Tensor,
    device: torch.device,
    half: bool,
    do_print: bool = False,
) -> Tuple[LvcEncode, LvcDecode]:
    norm = "ortho"

    references = None
    results = None
    grouping = "vertical_uv"
    yuv_mode = 444

    if dct_size is None:
        dct_layers = (
            DctYuvMetadata(half, norm),
            ChunkSplit(dct_size, do_print=do_print),
        )
        idct_layers = (
            ChunkCombine(yuv_mode, device, dct_size, is_half=half, do_print=do_print),
            IdctYuvMetadata(half, norm),
        )
    else:
        # grouping = lvc_params["grouping"]
        dct_layers = (
            ChunkSplit(dct_size, do_print=do_print),
            DctBlock(
                dct_size, yuv_mode, grouping, is_half=half, norm=norm, do_print=do_print
            ),
        )
        idct_layers = (
            IdctBlock(
                dct_size, yuv_mode, grouping, is_half=half, norm=norm, do_print=do_print
            ),
            ChunkCombine(yuv_mode, device, dct_size, is_half=half, do_print=do_print),
        )

    dct_encoder = LvcEncode(
        [
            Pad(yuv_mode, dct_size, do_print=do_print),
            RgbToYcbcrMetadata(w=w_yuv),
            # Downsample(mode=yuv_mode),
            # SubtractMean(device),
            dct_layers[0],
            dct_layers[1],
        ],
        references,
        results,
        chunk_size,
    ).to(device, non_blocking=True)

    dct_decoder = LvcDecode(
        [
            idct_layers[0],
            idct_layers[1],
            # RestoreMean(),
            # Upsample(mode=yuv_mode),
            YcbcrToRgbMetadata(w=w_yuv),
            Crop(do_print=do_print),
        ]
    ).to(device, non_blocking=True)

    if half:
        dct_encoder.half()
        dct_decoder.half()

    return (dct_encoder, dct_decoder)


def run_yuv420(
    crs: List[float],
    csnr_dbs: List[int],
    yuv_images: List[YuvImage],
    chunk_size: Tuple[int, int] | int,
    estimator: str,
) -> dict:
    norm = "ortho"
    power = 1.0
    yuv_mode = 420
    device = set_device()

    references = None
    results = None
    do_print = True

    res = {}

    for cr, csnr_db in itertools.product(crs, csnr_dbs):
        if do_print:
            print("-- cr: {}, csnr: {} dB".format(cr, csnr_db))

        if estimator == "zf":
            estimator_module = ZfEstimate(power)
        elif estimator == "llse":
            estimator_module = LlseEstimate(power)
        else:
            raise ValueError(
                "Wrong estimator: {}, choose 'zf' or 'llse'".format(estimator)
            )

        lvc_encoder = LvcEncode(
            [
                SubtractMean(device),
                DctYuvMetadata(is_half=False, norm=norm),
                ChunkSplit(dct_size=None),
                ChunkSelect(cr, do_print=do_print),
                PowerAllocate(power, do_print=do_print),
            ],
            references,
            results,
            chunk_size,
            do_print=do_print,
        )

        lvc_decoder = LvcDecode(
            [
                estimator_module,
                ChunkRestore(device=device, is_half=False),
                ChunkCombine(
                    yuv_mode, device, dct_size=None, is_half=False, do_print=do_print
                ),
                IdctYuvMetadata(is_half=False, norm=norm),
                RestoreMean(),
            ]
        )

        if csnr_db == "inf":
            lvc_chain = transforms.Compose([lvc_encoder, lvc_decoder])
        else:
            lvc_chain = transforms.Compose(
                [
                    lvc_encoder,
                    Channel(csnr_db, device, is_half=False, do_print=do_print),
                    lvc_decoder,
                ]
            )

        yuv_images_out = lvc_chain(yuv_images)

        psnr_yuv420 = []

        for inp, out in zip(yuv_images, yuv_images_out):
            diff_yuv420 = out.diff(inp)

            mse_yuv420 = torch.Tensor(
                [
                    diff_yuv420.y.square().mean(),
                    diff_yuv420.u.square().mean(),
                    diff_yuv420.v.square().mean(),
                ]
            )

            psnr_result = 10 * torch.log10(torch.div(1.0**2, mse_yuv420))
            psnr_yuv420.append(psnr_result)
            if do_print:
                print("  PSNR: ", psnr_result)

        res[(cr, csnr_db)] = psnr_yuv420

    return res


def run_rgb(
    crs: List[float],
    csnr_dbs: List[int],
    image_names: List[str],
    chunk_size: Tuple[int, int] | int,
) -> dict:
    norm = "ortho"
    power = 1.0
    yuv_mode = 420
    w_human = torch.Tensor([0.299, 0.587, 0.114])
    device = set_device()

    references = None
    results = None

    res = {}

    for cr, csnr_db in itertools.product(crs, csnr_dbs):
        res[(cr, csnr_db)] = {}

    for image_name in image_names:
        image_inp = Image.open(image_name).convert(mode="RGB")

        image_inp_tensor = transforms.ToTensor()(image_inp)
        image_inp_tensor = transforms.CenterCrop((288, 352))(image_inp_tensor)

        for cr, csnr_db in itertools.product(crs, csnr_dbs):
            lvc_encoder = LvcEncode(
                [
                    RgbToYcbcrMetadata(w=w_human),
                    Downsample(mode=yuv_mode),
                    SubtractMean(device),
                    DctYuvMetadata(is_half=False, norm=norm),
                    ChunkSplit(dct_size=None),
                    ChunkSelect(cr),
                    PowerAllocate(power),
                ],
                references,
                results,
                chunk_size,
            )

            lvc_decoder = LvcDecode(
                [
                    ZfEstimate(power),
                    ChunkRestore(device=device, is_half=False),
                    ChunkCombine(yuv_mode, device, dct_size=None, is_half=False),
                    IdctYuvMetadata(is_half=norm),
                    RestoreMean(),
                    Upsample(mode=yuv_mode),
                    YcbcrToRgbMetadata(w=w_human),
                ]
            )

            if csnr_db == "inf":
                lvc_chain = transforms.Compose([lvc_encoder, lvc_decoder])
            else:
                lvc_chain = transforms.Compose(
                    [lvc_encoder, Channel(csnr_db, device, is_half=False), lvc_decoder]
                )

            image_out_tensor = lvc_chain([image_inp_tensor])[0]

            mse = (image_out_tensor - image_inp_tensor).square().mean()
            psnr_total = 10 * torch.log10(1.0**2 / mse)

            res[(cr, csnr_db)][image_name.stem] = psnr_total

    return res


if __name__ == "__main__":
    #  image = Image.open("/home/kubouch/pictures/kodim/raw/kodim23.png")
    # image = Image.open("/home/kubouch/pictures/bc1_torture_16x16.png")
    # image = Image.open("C:/kubouch/data/kodim/raw/kodim23.png")
    # image = image.convert(mode="RGB")
    # image_size = image.size
    device = set_device()

    frame_mats = [
        sio.loadmat("reference/kodim23_cif_frame01.mat"),
        # sio.loadmat("reference/husky_cif_frame01.mat"),
    ]

    yuv_images_inp = [
        YuvImage(
            torch.Tensor(frame_mat["first_Y"]) / 255.0,
            torch.Tensor(frame_mat["first_U"]) / 255.0,
            torch.Tensor(frame_mat["first_V"]) / 255.0,
        )
        for frame_mat in frame_mats
    ]

    report_n = 0  # Which image to show and report
    image_size = yuv_images_inp[report_n].y.shape
    print("Image size: ", image_size)

    dct_mats = [
        sio.loadmat("reference/kodim23_cif_dct_gop01.mat"),
        sio.loadmat("reference/husky_cif_dct_gop01.mat"),
    ]

    mean_mats = [
        sio.loadmat("reference/kodim23_cif_mean_gop01_frame01.mat"),
        sio.loadmat("reference/husky_cif_mean_gop01_frame01.mat"),
    ]

    var_mats = [
        sio.loadmat("reference/kodim23_cif_var_gop01_frame01.mat"),
        sio.loadmat("reference/husky_cif_var_gop01_frame01.mat"),
    ]

    g_mats = [
        sio.loadmat("reference/kodim23_cif_g_gop01.mat"),
        sio.loadmat("reference/husky_cif_g_gop01.mat"),
    ]

    # Parameters
    w_human = torch.Tensor([0.299, 0.587, 0.114])
    norm = "ortho"
    chunk_size = 64  # (36, 44)  # h, w
    snr_db = 30
    cr = 1.0
    power = 1.0
    packet_loss = None  # "dc"
    yuv_mode = 420
    half = False
    do_print = True
    seed = 42
    dct_size = (8, 8)
    grouping = "vertical_uv"
    estimator_type = "zf"

    # Tracking results and reference values
    references = {
        "mean": [
            torch.Tensor(
                [
                    mean_mat["mean_y"][0][0] / 255.0,
                    mean_mat["mean_u"][0][0] / 255.0,
                    mean_mat["mean_v"][0][0] / 255.0,
                ]
            )
            for mean_mat in mean_mats
        ],
        "dct": [
            YuvImage(
                torch.Tensor(dct_mat["gop_dct_y"]) / 255.0,
                torch.Tensor(dct_mat["gop_dct_u"]) / 255.0,
                torch.Tensor(dct_mat["gop_dct_v"]) / 255.0,
            )
            for dct_mat in dct_mats
        ],
        "var": [
            torch.cat(
                [
                    torch.Tensor(var_mat["var_y"].reshape(-1)) / 255.0 / 255.0,
                    torch.Tensor(var_mat["var_u"].reshape(-1)) / 255.0 / 255.0,
                    torch.Tensor(var_mat["var_v"].reshape(-1)) / 255.0 / 255.0,
                ]
            )
            for var_mat in var_mats
        ],
        "g": [torch.Tensor(g_mat["Gn"]).diagonal() * 255.0 for g_mat in g_mats],
    }

    if references is None:
        results = {
            "mean": None,
            "dct": None,
            "var": None,
            "var_blockdct": None,
            "g": None,
        }
    else:
        results = {"mean": [], "dct": [], "var": [], "var_blockdct": [], "g": []}

    # Defining the LVC chain
    if estimator_type == "zf":
        estimator = ZfEstimate(power)
    else:
        estimator = LlseEstimate(power)

    if dct_size is None:
        dct_layers = (
            DctYuvMetadata(is_half=False, norm=norm, results=results["dct"]),
            ChunkSplit(dct_size),
        )

        idct_layers = (
            ChunkCombine(yuv_mode, device, dct_size, is_half=half),
            IdctYuvMetadata(half, norm),
        )
    else:
        dct_layers = (
            ChunkSplit(dct_size),
            DctBlock(
                dct_size,
                yuv_mode,
                grouping,
                is_half=half,
                norm=norm,
                do_print=do_print,
                results=results["var_blockdct"],
            ),
        )
        idct_layers = (
            IdctBlock(
                dct_size, yuv_mode, grouping, is_half=half, norm=norm, do_print=do_print
            ),
            ChunkCombine(
                yuv_mode,
                device=device,
                dct_size=dct_size,
                is_half=False,
                do_print=do_print,
            ),
        )

    lvc_encoder = LvcEncode(
        [
            # RgbToYcbcr(w_yuv),
            # Downsample(mode=yuv_mode),
            SubtractMean(device, results=results["mean"]),
            dct_layers[0],
            dct_layers[1],
            ChunkSelect(cr, results=results["var"], do_print=do_print),
            PowerAllocate(power, results=results["g"], do_print=do_print),
            RandomOrthogonal(seed, device, invert=False),
        ],
        references,
        results,
        chunk_size,
    )

    lvc_decoder = LvcDecode(
        [
            RandomOrthogonal(seed, device, invert=True),
            estimator,
            ChunkRestore(device=device, is_half=False, do_print=do_print),
            idct_layers[0],
            idct_layers[1],
            RestoreMean(),
            # Upsample(mode=yuv_mode),
            # YcbcrToRgb(w),
        ]
    )

    lvc_chain = transforms.Compose(
        [
            # transforms.ToTensor(),
            lvc_encoder,
            Channel(
                snr_db,
                device,
                is_half=False,
                packet_loss=packet_loss,
                seed=seed,
                do_print=True,
            ),
            lvc_decoder,
            # transforms.ToPILImage(),
        ]
    )

    yuv_images_out = lvc_chain(yuv_images_inp)
    yuv_image_out = yuv_images_out[report_n]

    y, u, v = yuv_image_out.y, yuv_image_out.u, yuv_image_out.v

    print("\n === REPORT === \n")
    lvc_encoder.report(report_n)

    print(
        "out min: ",
        (yuv_image_out.y.min(), yuv_image_out.u.min(), yuv_image_out.v.min()),
        "max: ",
        (yuv_image_out.y.max(), yuv_image_out.u.max(), yuv_image_out.v.max()),
    )

    diff_yuv420 = yuv_image_out.diff(yuv_images_inp[report_n])
    mse_yuv420 = torch.Tensor(
        [
            diff_yuv420.y.square().mean(),
            diff_yuv420.u.square().mean(),
            diff_yuv420.v.square().mean(),
        ]
    )
    print("MSE YUV 420: ", mse_yuv420)

    psnr_yuv420 = 10 * torch.log10(torch.div(1.0**2, mse_yuv420))
    print("PSNR YUV 420: {} dB".format(psnr_yuv420))

    inp_upsampled = YcbcrToRgbMetadata(w=w_human)(
        Upsample(mode=yuv_mode)(
            yuv_images_inp[report_n], Metadata(image_size, chunk_size)
        )[0],
        Metadata(image_size, chunk_size),
    )[0].clamp(0.0, 1.0)
    out_upsampled = YcbcrToRgbMetadata(w=w_human)(
        Upsample(mode=yuv_mode)(
            yuv_images_out[report_n], Metadata(image_size, chunk_size)
        )[0],
        Metadata(image_size, chunk_size),
    )[0].clamp(0.0, 1.0)

    to_tensor = transforms.ToTensor()
    mse = (out_upsampled - inp_upsampled).square().mean()
    psnr_total = 10 * torch.log10(1.0**2 / mse)
    print("PSNR total recontsructed: {} dB".format(psnr_total))

    # Plot images
    plt.close("all")

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Input image")
    axs = axs.flatten()
    axs[0].imshow(yuv_images_inp[report_n].y, cmap="gray")
    axs[0].set_title("Y")
    axs[1].imshow(yuv_images_inp[report_n].u, cmap="gray")
    axs[1].set_title("U")
    axs[2].imshow(yuv_images_inp[report_n].v, cmap="gray")
    axs[2].set_title("V")
    axs[3].imshow(transforms.ToPILImage()(inp_upsampled))
    axs[3].set_title("RGB")
    plt.tight_layout()

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Restored image")
    axs = axs.flatten()
    axs[0].imshow(yuv_image_out.y, cmap="gray")
    axs[0].set_title("Y")
    axs[1].imshow(yuv_image_out.u, cmap="gray")
    axs[1].set_title("U")
    axs[2].imshow(yuv_image_out.v, cmap="gray")
    axs[2].set_title("V")
    axs[3].imshow(transforms.ToPILImage()(out_upsampled))
    axs[3].set_title("RGB")
    plt.tight_layout()

    # Plot variances
    if dct_size is not None:
        print(
            "Variances of block-based DCT --",
            "Y:",
            results["var_blockdct"][report_n]["Y"].shape,
            ", U:",
            results["var_blockdct"][report_n]["U"].shape,
            ", V:",
            results["var_blockdct"][report_n]["V"].shape,
        )

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle("Variances")
    cmap = cm.jet
    axs = axs.flatten()
    if dct_size is None:
        var = results["var"][report_n][:64]
    else:
        var = results["var_blockdct"][report_n]["Y"]

    reshape_size_y = (8, 8)
    reshape_size_uv = (4, 4)

    axs[0].bar(range(var.size(dim=0)), var.sort(descending=True).values)
    axs[0].set_title("image[{}] Y variances".format(report_n))
    axs[0].set_yscale("log")
    im = axs[1].matshow(var.reshape(*reshape_size_y), cmap=cmap, norm=LogNorm())
    axs[1].set_title("image[{}] Y variances 2D".format(report_n))
    fig.colorbar(im, fraction=0.045, ax=axs[1])

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle("Variances")
    cmap = cm.jet
    axs = axs.flatten()
    if dct_size is None:
        var = results["var"][report_n][64:80]
    else:
        var = results["var_blockdct"][report_n]["U"]
    axs[0].bar(range(var.size(dim=0)), var.sort(descending=True).values)
    axs[0].set_title("image[{}] U variances".format(report_n))
    axs[0].set_yscale("log")
    im = axs[1].matshow(var.reshape(*reshape_size_uv), cmap=cmap, norm=LogNorm())
    axs[1].set_title("image[{}] U variances 2D".format(report_n))
    fig.colorbar(im, fraction=0.045, ax=axs[1])

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle("Variances")
    cmap = cm.jet
    axs = axs.flatten()
    if dct_size is None:
        var = results["var"][report_n][80:96]
    else:
        var = results["var_blockdct"][report_n]["V"]
    axs[0].bar(range(var.size(dim=0)), var.sort(descending=True).values)
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Input image")
    axs = axs.flatten()
    axs[0].imshow(yuv_images_inp[report_n].y, cmap="gray")
    axs[0].set_title("Y")
    axs[1].imshow(yuv_images_inp[report_n].u, cmap="gray")
    axs[1].set_title("U")
    axs[2].imshow(yuv_images_inp[report_n].v, cmap="gray")
    axs[2].set_title("V")
    axs[3].imshow(transforms.ToPILImage()(inp_upsampled))
    axs[3].set_title("RGB")
    plt.tight_layout()

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Restored image")
    axs = axs.flatten()
    axs[0].imshow(yuv_image_out.y, cmap="gray")
    axs[0].set_title("Y")
    axs[1].imshow(yuv_image_out.u, cmap="gray")
    axs[1].set_title("U")
    axs[2].imshow(yuv_image_out.v, cmap="gray")
    axs[2].set_title("V")
    axs[3].imshow(transforms.ToPILImage()(out_upsampled))
    axs[3].set_title("RGB")
    plt.tight_layout()

    # Plot variances
    if dct_size is not None:
        print(
            "Variances of block-based DCT --",
            "Y:",
            results["var_blockdct"][report_n]["Y"].shape,
            ", U:",
            results["var_blockdct"][report_n]["U"].shape,
            ", V:",
            results["var_blockdct"][report_n]["V"].shape,
        )

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle("Variances")
    cmap = cm.jet
    axs = axs.flatten()
    if dct_size is None:
        var = results["var"][report_n][:64]
    else:
        var = results["var_blockdct"][report_n]["Y"]
    axs[0].bar(range(var.size(dim=0)), var.sort(descending=True).values)
    axs[0].set_title("image[{}] Y variances".format(report_n))
    axs[0].set_yscale("log")
    im = axs[1].matshow(var.reshape(*reshape_size_y), cmap=cmap, norm=LogNorm())
    axs[1].set_title("image[{}] Y variances 2D".format(report_n))
    fig.colorbar(im, fraction=0.045, ax=axs[1])

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle("Variances")
    cmap = cm.jet
    axs = axs.flatten()
    if dct_size is None:
        var = results["var"][report_n][64:80]
    else:
        var = results["var_blockdct"][report_n]["U"]
    axs[0].bar(range(var.size(dim=0)), var.sort(descending=True).values)
    axs[0].set_title("image[{}] U variances".format(report_n))
    axs[0].set_yscale("log")
    im = axs[1].matshow(var.reshape(*reshape_size_uv), cmap=cmap, norm=LogNorm())
    axs[1].set_title("image[{}] U variances 2D".format(report_n))
    fig.colorbar(im, fraction=0.045, ax=axs[1])

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    fig.suptitle("Variances")
    cmap = cm.jet
    axs = axs.flatten()
    if dct_size is None:
        var = results["var"][report_n][80:96]
    else:
        var = results["var_blockdct"][report_n]["V"]
    axs[0].bar(range(var.size(dim=0)), var.sort(descending=True).values)
    axs[0].set_title("image[{}] V variances".format(report_n))
    axs[0].set_yscale("log")
    im = axs[1].matshow(var.reshape(*reshape_size_uv), cmap=cmap, norm=LogNorm())
    axs[1].set_title("image[{}] V variances 2D".format(report_n))
    fig.colorbar(im, fraction=0.045, ax=axs[1])

    plt.show()
