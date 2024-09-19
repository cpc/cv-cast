# Search for JPEG quality factor

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Literal, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor, to_pil_image

import torchjpeg.codec
from gradients.grace import get_quant_table_approx
from lvc.lvc import Metadata, SionnaChannel, SionnaConfig
from transforms.color_transforms import RgbToYcbcr, YcbcrToRgb
from transforms.metrics import mse_psnr
from utils import block_process, Size

COLOR_SAMP_FACTOR_VERTICAL = 1
COLOR_SAMP_FACTOR_HORIZONTAL = 1

CITYSCAPES_DIR = Path("/home/jakub/data/cityscapes/leftImg8bit/val")
CITYSCAPES_GLOB = "*/*.png"

COCO_DIR = Path("/home/jakub/data/coco/images/val2017")
COCO_GLOB = "*.jpg"

DATA = {
    "coco": {"dir": COCO_DIR, "glob": COCO_GLOB},
    "cityscapes": {"dir": CITYSCAPES_DIR, "glob": CITYSCAPES_GLOB},
}


class EncImgData(TypedDict):
    orig_size: float
    enc_size: float
    nsymbols: float  # number of symbols to send
    lct_cr: float  # LCT-equivalent compression ratio
    psnr: float
    mse: float


class EncDatasetData(TypedDict):
    mean_orig_size: float
    mean_enc_size: float
    mean_lct_cr: float
    mean_psnr: float


class ParamConfig(TypedDict):
    codec: Literal["jpeg", "grace"]
    init: int | float
    min: int | float
    max: int | float
    g_block_norm: Tensor | None
    W: Tensor | None
    valname: Literal["Q", "B"]
    fmt: str  # how to format the found parameter


def jpeg_round_fn(cr: float, target_cr: float) -> Callable:
    if cr > target_cr:
        return math.floor
    else:
        return math.ceil


def grace_round_fn(cr: float, target_cr: float) -> Callable:
    return lambda x: x


ROUND_FN = {
    "jpeg": jpeg_round_fn,
    "grace": grace_round_fn,
}


CMP_FN = {
    "jpeg": lambda cr, target_cr: cr > target_cr,  # increasing Q means decreasing size
    "grace": lambda cr, target_cr: cr < target_cr,  # increasing B means increasing size
}


def encode_jpeg(
    file: str | Path,
    param_config: ParamConfig,
    val: int | float,
    channel: SionnaChannel,
    save_dir: Path | None = None,
) -> EncImgData:
    codec = param_config["codec"]
    W = param_config["W"]

    if save_dir is not None:
        img_name_stem = Path(file).stem
        fmt_base = f"{img_name_stem}_{param_config['valname']}_{val}"

    img_rgb = to_tensor(Image.open(file).convert("RGB"))

    if img_rgb.shape[0] > 3:
        img_rgb = img_rgb[:3]

    num_orig_bytes = torch.prod(torch.tensor(img_rgb.shape)).item()

    if codec == "jpeg":
        Q = val
        # encode
        dimensions, quantization, Y_coefficients, CbCr_coefficients, enc_data = (
            torchjpeg.codec.quantize_at_quality(
                img_rgb,
                Q,
                COLOR_SAMP_FACTOR_VERTICAL,
                COLOR_SAMP_FACTOR_HORIZONTAL,
            )
        )
    elif codec == "grace":
        g_block_norm = param_config["g_block_norm"]
        B = val
        quant_tables = torch.zeros_like(g_block_norm).type(torch.int16)

        for i, g_norm in enumerate(g_block_norm):
            quant_tables[i], _ = get_quant_table_approx(g_norm, B)

        if save_dir is not None:
            qtable_file = save_dir / f"qtables_inp_{fmt_base}.png"
            to_pil_image(quant_tables.type(torch.uint8)).save(qtable_file)
            torch.save(quant_tables, qtable_file.with_suffix(".pt"))

        # RGB -> YUV
        rgb2yuv = RgbToYcbcr(W)
        img_yuv = rgb2yuv(img_rgb)

        # encode
        dimensions, quantization, Y_coefficients, CbCr_coefficients, enc_data = (
            torchjpeg.codec.quantize_at_quality_custom(
                img_yuv,
                100,
                quant_tables,
                COLOR_SAMP_FACTOR_VERTICAL,
                COLOR_SAMP_FACTOR_HORIZONTAL,
            )
        )

        if torch.all(quantization != quant_tables):
            print(f"ERROR: Image {file}, quantization tables do not match!")

    if save_dir is not None:
        qtable_file = save_dir / f"qtables_enc_{fmt_base}.png"
        to_pil_image(quantization.type(torch.uint8)).save(qtable_file)
        torch.save(quantization, qtable_file.with_suffix(".pt"))

    num_compressed_bytes = torch.prod(torch.tensor(enc_data.shape)).item()

    # decode
    if codec == "jpeg":
        raw = False
    elif codec == "grace":
        raw = True

    spatial = torchjpeg.codec.reconstruct_full_image(
        Y_coefficients, quantization, CbCr_coefficients, dimensions, raw=raw
    )

    if codec == "jpeg":
        spatial_rgb = spatial
    elif codec == "grace":
        # YUV -> RGB
        yuv2rgb = YcbcrToRgb(W)
        spatial_rgb = yuv2rgb(spatial).clamp(0.0, 1.0)

    mse, psnr = mse_psnr(img_rgb, spatial_rgb)

    nsymbols = math.ceil(
        num_compressed_bytes * 8 / channel.coderate / channel.nbits_per_sym
    )

    # not counting padding as we don't count padding in LCT either:
    cr = nmapped_to_cr(nsymbols, img_rgb.shape[-1], img_rgb.shape[-2])

    # print(
    #     f"orig_size: {num_orig_bytes:8.0f}, enc_size: {num_compressed_bytes:8.0f} bytes, nsymbols: {nsymbols:9.0f}, LCT CR: {cr:.5f}"
    # )

    # uncomment to double check:
    # # npadded: number of input symbols to the OFDM model (padded nsymbols)
    # # nmapped: number of symbols at the input of resource grid
    # # ntransmitted: number of symbols at the output of resource grid
    # nsymbols, npadded, nmapped, ntransmitted = get_nsymbols(enc_data, channel)
    # print(f"enc_size: {num_compressed_bytes:8.0f} bytes, nsymbols: {nsymbols:9.0f}, npadded: {npadded:9.0f}, nmapped: {nmapped: 9.0f}, ntransmitted: {ntransmitted:9.0f}, LCT CR: {cr:.5f}")

    if save_dir is not None:
        enc_file = save_dir / f"enc_{fmt_base}.jpg"
        out_file = save_dir / f"out_{fmt_base}.png"
        torchjpeg.codec.write_coefficients_custom(
            str(enc_file), dimensions, quantization, Y_coefficients, CbCr_coefficients
        )
        to_pil_image(spatial_rgb).save(out_file)

    return {
        "orig_size": num_orig_bytes,
        "enc_size": num_compressed_bytes,
        "nsymbols": nsymbols,
        "lct_cr": cr,
        "psnr": psnr,
        "mse": mse,
    }


def get_nsymbols(data: Tensor, channel: SionnaChannel) -> Tuple[int, int, int, int]:
    meta = Metadata((0, 0), (0, 0), (0, 0))
    bits = np.unpackbits(np.frombuffer(data.numpy(), dtype=np.uint8)).astype(np.float32)

    _ = channel([(torch.tensor(bits), meta)])

    return (
        channel.num_symbols,
        channel.num_padded,
        np.prod(channel.model.num_mapped_symbols),
        np.prod(channel.model.num_transmitted_symbols),
    )


def compress_dataset(
    dir: Path,
    glob: str,
    param_config: ParamConfig,
    val: int | float,
    nbits_per_sym: int,
    nimages: int | None = None,
    save_dir: Path | None = None,
    num_sionna_threads: int | None = 1,
) -> EncDatasetData:
    orig_sizes = []
    enc_sizes = []
    crs = []
    psnrs = []

    sionna_config: SionnaConfig = {"nbits_per_sym": nbits_per_sym, "coderate": 0.5}

    channel = SionnaChannel(
        100,
        "cpu",
        42,
        sionna_config,
        do_print=False,
        digital=True,
        dry=True,
        num_threads=num_sionna_threads,
    )

    for i, img in enumerate(dir.glob(glob)):
        if nimages is not None:
            if i >= nimages:
                break

        res = encode_jpeg(img, param_config, val, channel, save_dir)

        orig_sizes.append(res["orig_size"])
        enc_sizes.append(res["enc_size"])
        crs.append(res["lct_cr"])
        psnrs.append(res["psnr"])

    mean_orig_size = torch.tensor(orig_sizes, dtype=torch.float).mean().item()
    mean_enc_size = torch.tensor(enc_sizes, dtype=torch.float).mean().item()
    mean_cr = torch.tensor(crs, dtype=torch.float).mean().item()
    mean_psnr = torch.tensor(psnrs, dtype=torch.float).mean().item()

    return {
        "mean_orig_size": mean_orig_size,
        "mean_enc_size": mean_enc_size,
        "mean_lct_cr": mean_cr,
        "mean_psnr": mean_psnr,
    }


def nmapped_to_cr(nmapped: int, img_w: int, img_h: int, mode: int = 444) -> float:
    """Given a number of mapped symbols, what would be the equivalent
    compression ratio in LCT."""

    # number of source symbols: 2 symbols are transmitted per 1 channel use in LCT
    nsymbols = 2 * nmapped

    if mode == 444:
        ntotal = img_w * img_h * 3
    elif mode == 420:
        ntotal = img_w * img_h * 3 / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return nsymbols / ntotal


def find_val(
    target_cr: float,
    nbits_per_sym: int,
    param_config: ParamConfig,
    dataset: Literal["coco", "cityscapes"],
    nimages: int | None = None,
    max_iter: int = 100,
    save_dir: Path | None = None,
    do_print: bool = False,
    num_sionna_threads: int | None = 1,
) -> Tuple[int | float, float, float]:
    val = param_config["init"]
    max_val = param_config["max"]
    min_val = param_config["min"]
    codec = param_config["codec"]
    round_fn = ROUND_FN[codec]
    cmp_fn = CMP_FN[codec]
    data = DATA[dataset]
    eps_val = val * 1e-6
    eps_cr = target_cr * 0.025  # 2.5% of the CR

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    prev_enc_size = -1
    tested_val_cr = {}
    tested_val_psnr = {}
    i = 0
    while (val not in tested_val_cr) and (i < max_iter):
        res = compress_dataset(
            data["dir"],
            data["glob"],
            param_config,
            val,
            nbits_per_sym,
            nimages,
            save_dir=None,
            num_sionna_threads=num_sionna_threads,
        )
        cr = res["mean_lct_cr"]
        psnr = res["mean_psnr"]
        enc_size = res["mean_enc_size"]
        tested_val_cr[val] = cr
        tested_val_psnr[val] = psnr

        cmp_fn(cr, target_cr)

        if cmp_fn(cr, target_cr):
            max_val = val
        else:
            min_val = val

        round = round_fn(cr, target_cr)

        next_val = round((min_val + max_val) / 2)

        if do_print:
            qfmt = param_config["valname"] + ": " + param_config["fmt"].format(val)
            print(
                ", ".join(
                    [
                        qfmt,
                        f"next step: {next_val - val:12.5e}",
                        f'orig_size: {res["mean_orig_size"]:8.0f} B',
                        f"enc_size: {enc_size:8.0f} B",
                        f"LCT CR: {cr:.5f}",
                        f'PSNR: {res["mean_psnr"]:7.3f} dB',
                    ]
                )
            )

        if np.abs(next_val - val) < eps_val:
            if do_print:
                print(f"  stop: Param change < {eps_val}")
            val = next_val
            break

        val = next_val

        if (cr > target_cr) and (np.abs(cr - target_cr) < eps_cr):
            if do_print:
                print(f"  stop: CR change < {eps_cr}")
            break

        if prev_enc_size == enc_size:
            if do_print:
                print(f"  stop: Enc size same as previous")
            break

        prev_enc_size = enc_size
        i += 1

    # Try to get the CR that is higher than target CR
    sorted_by_val = dict(
        sorted(tested_val_cr.items(), key=lambda it: it[0], reverse=codec == "jpeg")
    )
    sorted_by_cr = dict(
        sorted(sorted_by_val.items(), key=lambda it: it[1], reverse=False)
    )

    res_val = None
    prev_cr = 0.0  # if the first value is already > target_cr, return it

    for param_val, cr in sorted_by_cr.items():
        if (cr == target_cr) or (cr > target_cr and prev_cr < target_cr):
            res_val = param_val
            break

        prev_cr = cr

    if res_val is None:
        # shouldn't happen, just in case:
        res_val = val

    if save_dir is not None:
        # save results only for the final chosen parameter
        _ = compress_dataset(
            data["dir"],
            data["glob"],
            param_config,
            res_val,
            nbits_per_sym,
            nimages,
            save_dir=save_dir,
        )

    return res_val, tested_val_cr[res_val], tested_val_psnr[res_val]


def run_qsearch(
    run_id: int,
    ntotal: int,
    codec: Literal["jpeg", "grace"],
    model: Literal[
        "fastseg_small", "fastseg_large", "yolov8_n", "yolov8_s", "yolov8_l"
    ],
    mode: Literal[444, 420],
    dist: Literal["dist_abs", "dist_sq"],
    norm: Literal["abs", "sq"],
    sub: Literal["submean", "nosubmean"],
    dctsz: Literal["ff", "bb8x8"],
    sc: Literal[1, 255],
    probe_dir: Path,
    nbits_per_sym: Literal[2, 4, 6],
    target_cr: float,
    nimages: int | None = None,
    dct_size: Size = Size(8, 8),
    outdir: Path | None = None,
    do_print: bool = False,
    num_sionna_threads: int | None = 1,
) -> dict | None:
    start_time = time.perf_counter()

    DATASET = {
        "fastseg_small": "cityscapes",
        "fastseg_large": "cityscapes",
        "yolov8_n": "coco",
        "yolov8_s": "coco",
        "yolov8_l": "coco",
    }

    msg = f"{run_id:4}/{ntotal:4}: " + ",".join(
        [
            f"{codec:5s}",
            f"{model:12s}",
            f"{mode:3d}",
            f"{dist:8s}",
            f"norm{norm:3s}",
            f"{sub:9s}",
            f"{dctsz:5s}",
            f"sc{sc:3d}",
            f"{2**nbits_per_sym:2d}-QAM",
            f"tgt{target_cr:7.5f}",
        ]
    )

    print(f"-- {msg}         (start) --")

    probefmt = f"{model}_{mode}_{dist}_{sub}_{dctsz}_sc{sc}_norm{norm}"
    probefile = probe_dir / f"probe_result_full_{probefmt}.pt"

    try:
        probe_results = torch.load(probefile)
    except FileNotFoundError:
        print(f"- file not found: {probefile}, skipping")
        return None

    W = probe_results["W"]
    # print(f"- W: {W}")

    # g = probe_results["grads_yuv"]
    # print(f"- g: {g.shape}, min: {g.min()}, max: {g.max()}")

    g_block_norm = probe_results["grads_norm"][dct_size.w * dct_size.h]

    # Duplicated code from Model.run_probe:
    # if dctsz == "ff":
    #     norm_abs = lambda tensor: tensor.abs().mean()
    #     norm_sq = lambda tensor: tensor.square().sum().sqrt()

    #     if norm == "abs":
    #         norm_fn = norm_abs
    #     elif norm == "sq":
    #         norm_fn = norm_sq

    #     # average DCT/grad obtained with full-frame DCT
    #     g_block_norm = block_process(
    #         g, (int(g.shape[1] / dct_size.h), int(g.shape[2] / dct_size.w)), norm_fn
    #     )
    # else:
    #     # use this for DCT/grad obtained with block DCT
    #     sz = math.ceil(g.shape[0] / 3)
    #     if norm == "abs":
    #         g_block_norm = torch.stack([x.abs().mean(dim=0) for x in g.split(sz)])
    #     elif norm == "sq":
    #         g_block_norm = torch.stack(
    #             [x.square().sum(dim=0).sqrt() for x in g.split(sz)]
    #         )

    # print(f"- g_block_norm: {g_block_norm.shape}")
    assert g_block_norm.shape[1] == dct_size.h
    assert g_block_norm.shape[2] == dct_size.w

    # Choose initial B such that with approximate quantization table max. q is 255.
    # This gives a reasonable initial ballpark value.
    B = 255 * g_block_norm.min().item() * dct_size.w * dct_size.h / 2

    # Run the search
    if codec == "jpeg":
        params: ParamConfig = {
            "codec": "jpeg",
            "init": 50,
            "min": 1,
            "max": 100,
            "g_block_norm": None,
            "W": None,
            "valname": "Q",
            "fmt": "{:3}",
        }
    elif codec == "grace":
        params: ParamConfig = {
            "codec": "grace",
            "init": B,
            "min": B / 255,
            "max": B * 255,
            "g_block_norm": g_block_norm,
            "W": W,
            "valname": "B",
            "fmt": "{:12.5e}",
        }

    dataset = DATASET[model]

    # print(
    #     f"- {params['valname']} init: {params['fmt'].format(params['init'])}, min: {params['fmt'].format(params['min'])}, max: {params['fmt'].format(params['max'])}"
    # )

    # for nbits_per_sym, target_cr in itertools.product(NBITS_PER_SYM, TARGET_CRS):
    q, cr, psnr = find_val(
        target_cr,
        nbits_per_sym,
        params,
        dataset,
        nimages=nimages,
        save_dir=None,
        do_print=do_print,
        num_sionna_threads=num_sionna_threads,
    )

    partial_res = {
        "codec": codec,
        "model": model,
        "mode": mode,
        "dist": dist,
        "norm": norm,
        "sub": sub,
        "dctsz": dctsz,
        "sc": sc,
        "dataset": dataset,
        "param_name": params["valname"],
        "param_init": params["init"],
        "nbits_per_sym": nbits_per_sym,
        "target_cr": target_cr,
        "param": q,
        "cr": cr,
        "psnr": psnr,
    }

    # print(
    #     f"  {2**nbits_per_sym:2}-QAM, target LCT CR: {target_cr:.5f}, actual LCT CR: {cr:.5f}, {params['valname']}: {params['fmt'].format(q)}, {psnr:7.3f} dB PSNR"
    # )

    duration = int(time.perf_counter() - start_time)
    duration_msg = f"{duration // 60:3d}m{duration % 60:02d}s"

    if outdir is None:
        print(f"== {msg} {duration_msg} (end)   ==")
    else:
        save_dir = outdir / "partial"
        save_dir.mkdir(exist_ok=True, parents=True)
        fname = (
            save_dir
            / f"{codec}_{model}_{mode}_{dist}_norm{norm}_{sub}_{dctsz}_sc{sc}_{nbits_per_sym}bits_tgt{target_cr:.5f}.pt"
        )
        print(f"== {msg} {duration_msg} (end) -> {fname} ==")
        torch.save(partial_res, fname)
        with open(fname.with_suffix(".json"), "w") as wf:
            json.dump(partial_res, wf)

    return partial_res


class QSearchFilters(TypedDict):
    mode: int
    dist: Literal["dist_sq", "dist_abs"]
    norm: Literal["sq", "abs"]
    sub: Literal["submean", "nosubmean"]
    dctsz: Literal["ff", "bb8x8", "bb16x16"]
    sc: float


DEFAULT_FILTERS: QSearchFilters = {
    "mode": 444,
    "dist": "dist_sq",
    "norm": "sq",
    "sub": "submean",
    "dctsz": "bb8x8",
    "sc": 1.0,
}


def fetch_param(
    data: pd.DataFrame,
    model: str,
    codec: str,
    nbits_per_sym: int,
    target_cr: float,
    filters: QSearchFilters,
) -> float | int | None:
    out = data

    all_filters = {
        "codec": codec,
        "model": model,
        "nbits_per_sym": nbits_per_sym,
        "target_cr": target_cr,
    }
    all_filters.update(filters)

    for name, val in all_filters.items():
        out = out[out[name] == val]

    if len(out) == 0:
        return None

    if len(out) > 1:
        raise ValueError(f"Got multiple results:\n{out}")

    out_param = out["param"].iloc[0]

    if codec == "jpeg":
        return int(out_param)
    elif codec == "grace":
        return float(out_param)
