"""Code related to implementing GRACE"""

from typing import Tuple

import torch
from torch import Tensor

# from models.model import plot_channels
from utils import Size


def get_k(B: float, theta: Tensor, theta_idx: Tensor, do_print: bool = False) -> int:
    """Get K given loss bound B and sorted max. loss increase theta"""

    K = 0

    for k, (th, th_idx) in reversed(list(enumerate(zip(theta, theta_idx)))):
        if k == 0:
            break

        theta_sum = theta[k:].sum()
        dk = (B - theta_sum) / k  # worst-case actual loss increase

        th_next = theta[k - 1]
        if do_print:
            print(
                f"k: {k:3}, thidx: {th_idx:3}, th: {th}, dk: {dk}, th_next: {th_next}"
            )

        if th < dk <= th_next:
            K = k
            break

    return K


def get_quant_table(
    g_norm: Tensor,
    dct_norm: Tensor,
    B: float,
    do_print: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate quantization table according to Algorithm 1"""

    assert g_norm.shape == dct_norm.shape

    max_loss_increase = g_norm * dct_norm

    # if do_show:
    #     plot_channels(
    #         max_loss_increase.cpu().detach().numpy(),
    #         "max loss increase",
    #         width=500,
    #         height=1000,
    #     )

    theta, theta_idx = max_loss_increase.view(-1).sort(descending=True)

    max_B = theta.sum()

    if do_print:
        print(f"max_loss_increase: {max_loss_increase.shape}")
        print(max_loss_increase)
        print(f"theta_idx: {theta_idx}")
        print(f"Max total loss increase, B <= {max_B}")

    if B > max_B:
        raise ValueError(f"Exceeded max. B ({max_B}): got {B}")

    if do_print:
        print(f"B: {B}, B/N: {B / 64}")

    K = get_k(B, theta, theta_idx, do_print=do_print)

    if do_print:
        print(f"K: {K}")

    max_theta = (B - theta[K:].sum()) / K
    d_idx = theta_idx[:K]
    d = max_loss_increase.clone()  # worst-case actual loss increase
    d.view(-1)[d_idx] = max_theta

    # just for printing:
    bounded = torch.full(d.shape, False)
    bounded.view(-1)[d_idx] = True

    q = 2 * d / g_norm

    return q, d, bounded, max_loss_increase


def get_quant_table_approx(g_norm: Tensor, B: float) -> Tuple[Tensor, Tensor]:
    """Calculate approximate quantization table ignoring constraint (5c)

    Same as Xiao et al., DNN-Driven Compressive Offloading for Edge-Assisted
    Semantic Video Segmentation, IEEE INFOCOM 2022.
    """

    N = torch.prod(torch.tensor(g_norm.shape)).item()
    d = torch.full(g_norm.shape, B / N, device=g_norm.device)
    q = 2 * d / g_norm
    q = q.round().clamp(1, 255)

    return q, d
