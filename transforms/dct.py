import math

import torch
import torchvision.transforms as T
from torch import Tensor
from torch.fft import fft, ifft
from torch.nn import ReplicationPad2d, Module

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from scipy.fft import dctn
from scipy.fft import idctn

# Notes:
# - JPEG uses DCT type 2 and IDCT type 2 (tha latter equals to DCT type 3)
# - 12x8 image is read into tensor with size [3, 8, 12]


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    Adapted from:
    https://github.com/zh217/torch-dct

    This version uses the torch.fft module and allows backpropagation

    Scipy DCT reference (the `norm` argument explained):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Adapted from:
    https://github.com/zh217/torch-dct

    This version uses the torch.fft module and allows backpropagation

    Scipy DCT reference (the `norm` argument explained):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
    """

    X_shape = X.shape
    N = X_shape[-1]

    X_v = X.contiguous().view(-1, X_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(X_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = ifft(V, dim=1).real

    # math.ceil makes idx1 and idx2 the same length if N is odd. Similarly,
    # the n and [:N] below ensures the last duplicated sample is dropped to
    # match the initial size.
    idx1 = torch.arange(math.ceil(N / 2), device=X.device)
    idx2 = torch.arange(N - 1, N // 2 - 1, -1, device=X.device)
    n = N if N % 2 == 0 else N + 1
    idx = torch.stack([idx1, idx2], dim=1).view(n)[:N]

    x = torch.index_select(v, 1, idx)

    return x.view(*X_shape)


def dct_2d(x, norm=None):
    """2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    Adapted from:
    https://github.com/zh217/torch-dct

    This version uses the torch.fft module and allows backpropagation

    Scipy DCT reference (the `norm` argument explained):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
    """

    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Adapted from:
    https://github.com/zh217/torch-dct

    This version uses the torch.fft module and allows backpropagation

    Scipy DCT reference (the `norm` argument explained):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
    """

    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_2d_block(x, norm=None, w=None, h=None):
    """Compute block-based DCT type II

    The image is segmented into w x h blocks (zero-padded if necessary) and DCT
    is performed for each block separately, then averaged over all blocks.
    The result is therefore of a size w x h.

    If w or h are not defined, the block size is inferred from X (=> the whole
    image is one block).
    """

    if len(x.shape) != 4:
        raise ValueError("Input must be 4D image batch (B, C, H, W)")

    # Input tensor size
    (B, C, H, W) = x.shape

    # Infer block size if not specified
    if not h:
        h = H

    if not w:
        w = W

    # Padding by replicating border
    if ((H % h) != 0) or ((W % w) != 0):
        h_pad = H % h
        w_pad = W % w
        padding = ReplicationPad2d((0, w_pad, 0, h_pad))
        x = padding(x)

    # Split 2nd and 3rd dimensions into w x h blocks (2nd dim is no. of blocks)
    x_blocks = x.unfold(2, h, w).unfold(3, h, w)
    x_blocks = x_blocks.contiguous().view(-1, B, C, h, w)

    # DCT for each block
    X_blocks = dct_2d(x_blocks, norm=norm)

    # Average the blocks
    return X_blocks


def idct_2d_block(X, norm=None, w=None, h=None):
    """Compute block-based IDCT type II

    The image is segmented into w x h blocks (zero-padded if necessary) and DCT
    is performed for each block separately, then averaged over all blocks.
    The result is therefore of a size w x h.

    If w or h are not defined, the block size is inferred from X (=> the whole
    image is one block).
    """

    if len(X.shape) != 4:
        raise ValueError("Input must be 4D image batch (B, C, H, W)")

    # Input tensor size
    (B, C, H, W) = X.shape

    # Infer block size if not specified
    if not h:
        h = H

    if not w:
        w = W

    # Padding by replicating border
    if ((H % h) != 0) or ((W % w) != 0):
        h_pad = H % h
        w_pad = W % w
        padding = ReplicationPad2d((0, w_pad, 0, h_pad))
        X = padding(X)

    # Split 2nd and 3rd dimensions into w x h blocks (2nd dim is no. of blocks)
    X_blocks = X.unfold(2, h, w).unfold(3, h, w)
    X_blocks = X_blocks.contiguous().view(-1, B, C, h, w)

    # DCT for each block
    x_blocks = idct_2d(X_blocks, norm=norm)

    # Average the blocks
    return x_blocks


class Dct(Module):
    def __init__(self, norm: str = "ortho"):
        super().__init__()
        self.norm = norm

    def forward(self, inp: Tensor) -> Tensor:
        return dct_2d(inp, norm=self.norm)


class Idct(Module):
    def __init__(self, norm: str = "ortho"):
        super().__init__()
        self.norm = norm

    def forward(self, inp: Tensor) -> Tensor:
        return idct_2d(inp, norm=self.norm)


def test_dct(input_image, show=True, norm=None):
    """Compare scipy vs pytorch DCT (both computed on CPU)"""

    print("# DCT test, norm:", norm)

    preprocess = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            #  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = preprocess(input_image)
    img_tensor.requires_grad_(True)
    img_tensor.retain_grad()
    print("# Image tensor size:", img_tensor.size())

    img_numpy = img_tensor.cpu().detach().numpy()

    # Pytorch DCT + IDCT (pytorch tensor data)
    dct_torch_rgb = dct_2d(img_tensor, norm=norm)
    idct_torch_rgb = idct_2d(dct_torch_rgb, norm=norm)

    # Scipy DCT + IDCT (numpy data)
    dct_scipy_rgb = torch.empty_like(img_tensor)
    idct_scipy_rgb = torch.empty_like(img_tensor)
    for ch in range(3):
        dct_scipy = dctn(img_numpy[ch], norm=norm, type=2)
        idct_scipy = idctn(dct_scipy, norm=norm, type=2)
        dct_scipy_rgb[ch] = torch.from_numpy(dct_scipy)
        idct_scipy_rgb[ch] = torch.from_numpy(idct_scipy)

    dct_torch_rgb.requires_grad_(True)
    dct_torch_rgb.retain_grad()
    idct_torch_rgb.requires_grad_(True)
    idct_torch_rgb.retain_grad()

    out = idct_torch_rgb.sum()
    out.backward()
    print("# Image tensor has gradient:", (img_tensor.grad != None))

    print("# Diff abs DCT vs scipy:")
    print("#  torch: ", torch.abs(dct_scipy_rgb - dct_torch_rgb).sum())
    print("# Diff abs IDCT vs scipy:")
    print("#  torch: ", torch.abs(idct_scipy_rgb - idct_torch_rgb).sum())
    print("# Diff abs IDCT vs raw")
    print("#  scipy: ", torch.abs(img_tensor - idct_scipy_rgb).sum())
    print("#  torch: ", torch.abs(img_tensor - idct_torch_rgb).sum())

    if show:
        dct_scipy_mag = dct_scipy_rgb.square().sum(dim=0).sqrt()
        dct_torch_mag = dct_torch_rgb.square().sum(dim=0).sqrt().detach()

        restored_img_scipy = idct_scipy_rgb.permute(1, 2, 0)
        restored_img_torch = idct_torch_rgb.permute(1, 2, 0)

        dct_sz_x = img_tensor.shape[-1]
        dct_sz_y = img_tensor.shape[-2]

        plt.close("all")
        cmap = cm.jet
        plt.subplot(221)
        plt.imshow(dct_scipy_mag.log10().numpy()[:dct_sz_x, :dct_sz_y], cmap=cmap)
        plt.colorbar(fraction=0.045)
        plt.title("log10 DCT magnitude scipy")
        plt.subplot(222)
        plt.imshow(restored_img_scipy.numpy())
        plt.title("restored image scipy")

        plt.subplot(223)
        plt.imshow(dct_torch_mag.log10().numpy()[:dct_sz_x, :dct_sz_y], cmap=cmap)
        plt.colorbar(fraction=0.045)
        plt.title("log10 DCT magnitude torch")
        plt.subplot(224)
        plt.imshow(restored_img_torch.detach().numpy())
        plt.title("restored image torch")
        plt.show()
