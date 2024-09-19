import itertools
import math
import cProfile
import time
from pathlib import Path
from pstats import SortKey
from typing import Tuple, List, TypedDict, Collection, Literal, NotRequired

import matplotlib

try:
    matplotlib.use("TkAgg")
except:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pprofile
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.profiler import profile as torch_profile
from torch.profiler import record_function
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots

from lvc.lvc import (
    DownsampleTensor,
    UpsampleTensor,
    WrapMetadata,
    StripMetadata,
    Pad,
    Crop,
    RgbToYcbcrMetadata,
    YcbcrToRgbMetadata,
    TensorToImage,
    ImageToTensor,
    ChunkSplit,
    ChunkCombine,
)
from transforms.dct import Dct, Idct
from transforms.color_transforms import RgbToYcbcr, YcbcrToRgb
from transforms.metrics import mse_psnr, ssim, msssim
from transforms import Multiply, Divide
from utils import block_process, set_device


class ModelConfig(TypedDict):
    name: str
    variant: str
    snapshot: str | Path
    unit: str
    task: NotRequired[str | Path]
    data_file: NotRequired[str | Path]


def plot_channels(
    image: np.ndarray,
    label: str,
    show: bool = True,
    save: str | Path | None = None,
    log: bool = False,
    width: int = 800,
    height: int = 1600,
):
    fig = make_subplots(rows=3, cols=1)

    # https://community.plotly.com/t/how-to-set-log-scale-for-z-axis-on-a-heatmap/292/8
    def colorbar(nmin, nmax):
        labels = np.sort(
            np.concatenate(
                [
                    np.linspace(10**nmin, 10**nmax, 10),
                    10 ** np.linspace(nmin, nmax, 10),
                ]
            )
        )
        # vals = np.linspace(nmin, nmax, nmax+nmin+1)

        return dict(
            tick0=nmin,
            # title="Log Scale",
            tickmode="array",
            tickvals=np.log10(labels),
            ticktext=[f"{x:.2e}" for x in labels],
            # tickvals=vals,
            # ticktext=[f"{10**x:.2e}" for x in labels],
            # tickvals=np.linspace(nmin, nmax, nmax - nmin + 1),
            # ticktext=[
            #     f"{x:.0e}" for x in 10 ** np.linspace(nmin, nmax, nmax - nmin + 1)
            # ],
        )

    if log:
        zero_mask = np.logical_or(image == 0.0, image == np.nan)
        img_nz = image[~zero_mask]
        image[zero_mask] = np.nan

        gmin = img_nz.min()
        gmax = img_nz.max()
        nmin = int(np.floor(np.log10(gmin)))
        nmax = int(np.ceil(np.log10(gmax)))
        fig.add_trace(
            go.Heatmap(
                z=np.log10(image[0]),
                customdata=image[0],
                hovertemplate="x: %{x} <br>" + "y: %{y} <br>" + "z: %{customdata:.2e}",
                colorbar=colorbar(nmin, nmax),
                # colorscale="Inferno",
                # reversescale=True,
                zmin=np.log10(gmin),
                zmax=np.log10(gmax),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=np.log10(image[1]),
                customdata=image[1],
                hovertemplate="x: %{x} <br>" + "y: %{y} <br>" + "z: %{customdata:.2e}",
                colorbar=colorbar(nmin, nmax),
                # colorscale="Inferno",
                # reversescale=True,
                zmin=np.log10(gmin),
                zmax=np.log10(gmax),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=np.log10(image[2]),
                customdata=image[2],
                hovertemplate="x: %{x} <br>" + "y: %{y} <br>" + "z: %{customdata:.2e}",
                colorbar=colorbar(nmin, nmax),
                # colorscale="Inferno",
                # reversescale=True,
                zmin=np.log10(gmin),
                zmax=np.log10(gmax),
            ),
            row=3,
            col=1,
        )
    else:
        gmin = image.min()
        gmax = image.max()

        fig.add_trace(go.Heatmap(z=image[0], zmin=gmin, zmax=gmax), row=1, col=1)
        fig.add_trace(go.Heatmap(z=image[1], zmin=gmin, zmax=gmax), row=2, col=1)
        fig.add_trace(go.Heatmap(z=image[2], zmin=gmin, zmax=gmax), row=3, col=1)

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_layout(title=label, width=width, height=height)

    if save:
        fig.write_image(save, scale=2.0)
        fig.write_html(Path(save).with_suffix(".html"))

    if show:
        fig.show()


def plot_channels_old(
    image: np.ndarray,
    labels: List[str],
    figure: int,
    show: bool = True,
    save: str | Path | None = None,
    tight_layout: bool = True,
):
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    gmin = image.min()
    gmax = image.max()
    cmap = cm.jet

    plt.figure(figure)
    plt.subplot(311)
    plt.imshow(image[0], cmap=cmap, norm=plt.Normalize(gmin, gmax))
    plt.colorbar(fraction=0.045)
    plt.title(labels[0])

    plt.subplot(312)
    plt.imshow(image[1], cmap=cmap, norm=plt.Normalize(gmin, gmax))
    plt.colorbar(fraction=0.045)
    plt.title(labels[1])

    plt.subplot(313)
    plt.imshow(image[2], cmap=cmap, norm=plt.Normalize(gmin, gmax))
    plt.colorbar(fraction=0.045)
    plt.title(labels[2])

    if tight_layout:
        plt.tight_layout()

    if save is not None:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()


class Model:
    def __init__(
        self,
        config: ModelConfig,
        device: str | torch.device | None = None,
        num_batches: int | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        lvc_chain: T.Compose | None = None,
        color_space: str = "rgb",
        do_print: bool = False,
    ):
        self._config = config

        if color_space not in ["yuv", "rgb"]:
            raise ValueError(f"Unknown color space: {color_space}")
        self._color_space = color_space

        if not device:
            device = set_device(do_print=do_print)
        self._device = device

        self._name = config["name"]
        self._variant = config["variant"]
        self._snapshot = config["snapshot"]
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._lvc_chain = lvc_chain
        self._do_print = do_print

    def _get_dataloader(
        self, split: Literal["train", "val"]
    ) -> Collection[torch.Tensor]:
        raise NotImplementedError()

    def _get_batch_images(self, batch) -> torch.Tensor:
        raise NotImplementedError()

    def _probe(self, probe_config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def _get_params(self) -> List:
        raise NotImplementedError()

    def eval(self) -> dict:
        raise NotImplementedError()

    def bench(
        self,
        do_probe: bool = True,
        do_eval: bool = True,
        profile_mode: str | None = None,
    ):
        if profile_mode:
            profile_mode = profile_mode.lower()

        if do_probe:
            pass

        if do_eval:
            if profile_mode == "cprofile":
                with cProfile.Profile() as prof:
                    eval_start = time.perf_counter()
                    res = self.eval()
                    eval_end = time.perf_counter()

                    prof.print_stats(SortKey.TIME)
            if profile_mode == "pprofile":
                prof = pprofile.StatisticalProfile()
                with prof():
                    eval_start = time.perf_counter()
                    res = self.eval()
                    eval_end = time.perf_counter()

                fname = f"{self._name}_{self._variant}_eval.out"
                with open(fname, "w") as wf:
                    prof.callgrind(wf)
            elif profile_mode == "torch":
                with torch_profile(
                    record_shapes=True,
                ) as torch_prof:
                    with record_function(f"{self._name}_{self._variant}_eval"):
                        eval_start = time.perf_counter()
                        res = self.eval()
                        eval_end = time.perf_counter()

                print(
                    torch_prof.key_averages().table(
                        sort_by="cpu_time_total", row_limit=10
                    )
                )
            elif profile_mode == "cpu" or profile_mode is None:
                eval_start = time.perf_counter()
                res = self.eval()
                eval_end = time.perf_counter()
            else:
                raise ValueError(f"Unknown profile mode: {profile_mode}")

            eval_time = eval_end - eval_start
            if self._num_batches:
                print(
                    f"Eval: {eval_time:.4f} s ({eval_time / self._num_batches:.4f} s/batch,"
                    + f" {eval_time / self._num_batches / self._batch_size:.4f} s/image)"
                )
            else:
                print(f"Eval: {eval_time:.4f} s (unknown num_batches)")

            print(res)

    @property
    def img_w(self) -> int:
        raise NotImplementedError()

    @property
    def img_h(self) -> int:
        raise NotImplementedError()

    def chunk_size(self, nchunks: int) -> Tuple[int, int]:
        """Get chunk size for probing, given a target number of chunks.

        Returns chunk size as (height, width)
        """
        nchunks_xy = math.sqrt(nchunks)

        if nchunks_xy == int(nchunks_xy):
            nchunks_xy = int(nchunks_xy)
        else:
            raise ValueError(f"Invalid number of chunks: {nchunks}")

        if (self.img_w / nchunks_xy != int(self.img_w / nchunks_xy)) or (
            self.img_h / nchunks_xy != int(self.img_h / nchunks_xy)
        ):
            raise ValueError(
                f"Image size {self.img_w}x{self.img_h} not evenly divisible by number"
                + f" of chunks {nchunks_xy}"
            )

        return (int(self.img_h / nchunks_xy), int(self.img_w / nchunks_xy))

    def run_probe(
        self,
        grad_type: Literal["dist_sq", "dist_precise", "dist_abs"],
        possible_nchunks: List[int],
        dct_size: Tuple[int, int] | None = None,
        subtract_mean: bool = True,
        scale: float = 1.0,  # value to multiply RGB input with
        norm: Literal["abs", "sq"] = "sq",
    ) -> dict:
        result = {}

        if dct_size is None:
            # Running probe for use in LVCT
            preprocess_rgb = nn.ModuleList([Multiply(scale)])
            postprocess_rgb = nn.ModuleList([Divide(scale)])
        else:
            # Running probe with block-based DCT (used for GRACE reproduction)
            mode = 444
            chunk_size = dct_size[0] * dct_size[1]

            preprocess_rgb = nn.ModuleList(
                [
                    Multiply(scale),
                    WrapMetadata(chunk_size),
                    Pad(mode, dct_size),
                    TensorToImage(),
                    ChunkSplit(dct_size),
                ]
            )

            postprocess_rgb = nn.ModuleList(
                [
                    ChunkCombine(mode, self._device, dct_size, is_half=False),
                    ImageToTensor(),
                    Crop(),
                    StripMetadata(),
                    Divide(scale),
                ]
            )

        probe_config_rgb = {
            "dct": Dct(),
            "idct": Idct(),
            "dct_preprocess": preprocess_rgb,  # will be run before DCT
            "idct_postprocess": postprocess_rgb,  # will be run after IDCT
            "seed": 42,
            "num_batches": self._num_batches,
            "subtract_mean": subtract_mean,
        }

        res_dct, res_dct_grad = self._probe(probe_config_rgb)

        if grad_type == "dist_abs":
            mean_dct = res_dct.abs().mean(dim=0)
            mean_dct_grad = res_dct_grad.abs().mean(dim=0)
        else:
            mean_dct = res_dct.mean(dim=0)
            mean_dct_grad = res_dct_grad.mean(dim=0)

        result["grads_rgb"] = mean_dct_grad.clone().cpu()
        result["dct_rgb"] = mean_dct.clone().cpu()

        if self._do_print:
            print(
                "gradients: ",
                mean_dct_grad.shape,
                ", mean: ",
                torch.mean(mean_dct_grad, dim=(1, 2)),
            )
            sens_rgb = mean_dct_grad.square().sum(dim=(1, 2)).sqrt()
            sens_rgb = sens_rgb.div(sens_rgb.sum())
            print("\nSensitivity to RGB channels:", sens_rgb, ", sum:", sens_rgb.sum())

        # Compute YUV weights
        z1 = torch.median(mean_dct_grad[2] / mean_dct_grad[1])
        z2 = torch.median(mean_dct_grad[0] / mean_dct_grad[1])

        if self._do_print:
            print(f"z1: {z1}, z2: {z2}")

        Wr = z2 / (1 + z1 + z2)
        Wg = 1 / (1 + z1 + z2)
        Wb = z1 / (1 + z1 + z2)
        W = torch.Tensor((Wr, Wg, Wb))
        result["W"] = W

        # Human values (CCIR.601):
        # Wr = 0.299
        # Wg = 0.587
        # Wb = 0.114
        if self._do_print:
            W_human = torch.Tensor((0.299, 0.587, 0.114))
            print(
                "\nWr: {:.4f}, Wg: {:.4f}, Wb: {:.4f}, sum: {}".format(
                    Wr, Wg, Wb, Wr + Wg + Wb
                )
            )
            print("human:", W_human)

        if self._color_space == "yuv":
            if self._do_print:
                print("\nCalculating per-chunk norm of YUV sensitivity")

            if dct_size is None:
                preprocess_yuv = nn.ModuleList([Multiply(scale), RgbToYcbcr(W)])
                postprocess_yuv = nn.ModuleList([YcbcrToRgb(W), Divide(scale)])
            else:
                mode = 444
                chunk_size = dct_size[0] * dct_size[1]

                preprocess_yuv = nn.ModuleList(
                    [
                        Multiply(scale),
                        WrapMetadata(chunk_size),
                        Pad(mode, dct_size),
                        RgbToYcbcrMetadata(W),
                        ChunkSplit(dct_size),
                    ]
                )

                postprocess_yuv = nn.ModuleList(
                    [
                        ChunkCombine(mode, self._device, dct_size, is_half=False),
                        YcbcrToRgbMetadata(W),
                        Crop(),
                        StripMetadata(),
                        Divide(scale),
                    ]
                )

            probe_config_yuv = {
                "dct": Dct(),
                "idct": Idct(),
                "dct_preprocess": preprocess_yuv,  # will run before DCT:
                "idct_postprocess": postprocess_yuv,  # will run after IDCT:
                "seed": 42,
                "num_batches": self._num_batches,
                "subtract_mean": subtract_mean,
            }

            res_dct_yuv, res_dct_grad_yuv = self._probe(probe_config_yuv)

            if grad_type == "dist_abs":
                mean_dct_yuv = res_dct_yuv.abs().mean(dim=0)
                mean_dct_grad_yuv = res_dct_grad_yuv.abs().mean(dim=0)
            else:
                mean_dct_yuv = res_dct_yuv.mean(dim=0)
                mean_dct_grad_yuv = res_dct_grad_yuv.mean(dim=0)

            if dct_size is None:
                # nearest interpolation selects the upper-left of the four
                # downsampled coefficients
                mean_dct_grad_yuv_420 = DownsampleTensor(420, "nearest")(
                    mean_dct_grad_yuv
                ).squeeze()
            else:
                # downsampling doesn't work well with block-based DCT because
                # the tensor has many channels instead of 3
                mean_dct_grad_yuv_420 = torch.zeros_like(mean_dct_grad_yuv)

            result["grads_yuv"] = mean_dct_grad_yuv.clone().cpu()
            result["grads_yuv_420"] = mean_dct_grad_yuv_420.clone().cpu()
            result["dct_yuv"] = mean_dct_yuv.clone().cpu()

            if self._do_print:
                print(
                    "YUV gradients: ",
                    mean_dct_grad_yuv.shape,
                    ", mean: ",
                    torch.mean(mean_dct_grad_yuv, dim=(1, 2)),
                )
                sens_yuv = mean_dct_grad_yuv.square().sum(dim=(1, 2)).sqrt()
                sens_yuv = sens_yuv.div(sens_yuv.sum())
                print(
                    "\nSensitivity to YUV channels:", sens_yuv, ", sum:", sens_yuv.sum()
                )

        result["dct_var"] = {}
        result["grads_norm"] = {}
        result["grads_norm_420"] = {}

        if dct_size is None:
            if norm == "abs":
                norm_fn = lambda tensor: tensor.abs().mean()
            elif norm == "sq":
                norm_fn = lambda tensor: tensor.square().sum().sqrt()

            var_fn = torch.var

            for nchunks in possible_nchunks:
                chunk_size = self.chunk_size(nchunks)

                dct = result[f"dct_{self._color_space}"]
                result["dct_var"][nchunks] = (
                    block_process(dct, chunk_size, var_fn).detach().clone().cpu()
                )

                grads = result[f"grads_{self._color_space}"]
                result["grads_norm"][nchunks] = (
                    block_process(grads, chunk_size, norm_fn).detach().clone().cpu()
                )

                if self._color_space == "yuv":
                    grads_420 = result[f"grads_yuv_420"]
                    result["grads_norm_420"][nchunks] = (
                        block_process(grads_420, chunk_size, norm_fn)
                        .detach()
                        .clone()
                        .cpu()
                    )
        else:
            dct = result[f"dct_{self._color_space}"]
            grads = result[f"grads_{self._color_space}"]
            if self._color_space == "yuv":
                grads_420 = result[f"grads_yuv_420"]

            dsz = math.ceil(dct.shape[0] / 3)
            gsz = math.ceil(grads.shape[0] / 3)

            if norm == "abs":
                dct_block_norm = torch.stack(
                    [x.abs().mean(dim=0) for x in dct.split(dsz)]
                )
                g_block_norm = torch.stack(
                    [x.abs().mean(dim=0) for x in grads.split(gsz)]
                )
                if self._color_space == "yuv":
                    g_block_norm_420 = torch.stack(
                        [x.abs().mean(dim=0) for x in grads_420.split(gsz)]
                    )
            elif norm == "sq":
                dct_block_norm = torch.stack(
                    [x.square().sum(dim=0).sqrt() for x in dct.split(dsz)]
                )
                g_block_norm = torch.stack(
                    [x.square().sum(dim=0).sqrt() for x in grads.split(gsz)]
                )
                if self._color_space == "yuv":
                    g_block_norm_420 = torch.stack(
                        [x.square().sum(dim=0).sqrt() for x in grads_420.split(gsz)]
                    )

            nchunks = dct_size[0] * dct_size[1]
            result["dct_var"][nchunks] = dct_block_norm
            result["grads_norm"][nchunks] = g_block_norm
            if self._color_space == "yuv":
                result["grads_norm_420"][nchunks] = g_block_norm_420

        return result

    def eval_metrics(self, ref_model, do_print: bool = False) -> dict:
        dataloader = self._get_dataloader("val")
        ref_dataloader = ref_model._get_dataloader("val")

        if len(dataloader) != len(ref_dataloader):
            raise ValueError("Dataloaders must have the same length")

        total_num_batches = len(dataloader)

        if self._num_batches is None:
            self._num_batches = total_num_batches
            batch_iter = dataloader
            ref_batch_iter = ref_dataloader
        else:
            batch_iter = itertools.islice(dataloader, self._num_batches)
            ref_batch_iter = itertools.islice(ref_dataloader, self._num_batches)

        batch_iter = map(self._get_batch_images, batch_iter)
        ref_batch_iter = map(self._get_batch_images, ref_batch_iter)

        mses = []
        psnrs = []
        ssims = []
        msssims = []

        for batch_id, (images, ref_images) in enumerate(
            zip(batch_iter, ref_batch_iter)
        ):
            if do_print:
                print(
                    f"\rbatch {batch_id+1}/{self._num_batches}: data {images.shape} {ref_images.shape}, min: {images.min()} {ref_images.min()}, max: {images.max()} {ref_images.max()}",
                    end="",
                    flush=True,
                )

            for img, ref_img in zip(images, ref_images):
                mse_val, psnr_val = mse_psnr(img, ref_img)
                mses.append(mse_val)
                psnrs.append(psnr_val)

            for ssim_val in ssim(images, ref_images):
                ssims.append(ssim_val.item())

            for msssim_val in msssim(images, ref_images):
                msssims.append(msssim_val.item())

        if do_print:
            print("")

        return {
            "mse": {"mean": np.mean(mses), "std": np.std(mses)},
            "psnr": {"mean": np.mean(psnrs), "std": np.std(psnrs)},
            "ssim": {"mean": np.mean(ssims), "std": np.std(ssims)},
            "msssim": {"mean": np.mean(msssims), "std": np.std(msssims)},
        }

    def get_num_params(self) -> int:
        """https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/4"""

        pp = 0

        for p in self._get_params():
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn

        return pp
