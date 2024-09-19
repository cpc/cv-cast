#!/usr/bin/env python3

from pathlib import Path

import sys

import numpy as np
import matplotlib.font_manager as fm

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont


FSS_DIR = "/home/zadnik/git/nn-spectral-sensitivity/experiments/runs"
# Y7_DIR = "C:/kubouch/data/cv_results/yolov7-images"


def get_lvc_string(lvc):
    if lvc == "raw":
        return "raw"
    else:
        return "lvc-" + "_".join(
            [
                "loss" + lvc["packet_loss"],
                "nchunks" + lvc["nchunks"],
                "dct" + lvc["grouping"],
                lvc["estimator"],
                "csnr" + lvc["csnr_db"] + "db",
                "cr" + lvc["cr"],
            ]
        )


def dirs(base_dir: str, train_str: str, eval_str: str):
    return Path(base_dir) / train_str / eval_str


def generate_image(
    filename,
    versions,
    ref_bases,
    crop_coords,
    crop_sizes,
    size=None,
    filter=Image.Resampling.BICUBIC,
    text_fill=(255, 255, 255),
    font_ratio=0.1,
):
    if size is None:
        size_x = crop_sizes[0][0]
        size_y = crop_sizes[0][1]
    else:
        size_x = size[0]
        size_y = size[1]

    w = size_x * len(versions[0])  # *(2 + len(versions))
    h = size_y * len(versions)  # *len(ref_bases)
    canvas_size = (w, h)
    # font = ImageFont.truetype('DejaVuSerif.ttf', 17)
    root = tk.Tk()
    fonts = list(set([f.name for f in fm.fontManager.ttflist]))
    fonts.sort()
    combo = ttk.Combobox(root, value=fonts)
    combo.pack()
    fontsize = round(font_ratio * size_y)
    font = ImageFont.truetype(
        fm.findfont(fm.FontProperties(family=combo.get())), fontsize
    )
    canvas = Image.new("RGB", canvas_size)

    for i, (ref_base, (crop_ox, crop_oy), (crop_dx, crop_dy)) in enumerate(
        zip(ref_bases, crop_coords, crop_sizes)
    ):
        crop_coords = (crop_ox, crop_oy, crop_ox + crop_dx, crop_oy + crop_dy)

        for j, version in enumerate(versions[i]):
            print("version:     ", version)
            base_dir, train_dir, eval_dir, prefix, label = version
            img_dir = dirs(base_dir, train_dir, eval_dir)
            img = Path(img_dir) / (prefix + ref_base)
            print("  img_path:  ", img)
            print("  exists:    ", img.exists())
            img_crop = Image.open(img).crop(crop_coords)
            if size_x != crop_dx or size_y != crop_dy:
                img_crop = img_crop.resize((size_x, size_y), resample=filter)
            text_w = font.getsize(label)[0]
            text_draw = ImageDraw.Draw(img_crop)
            text_draw.text(
                (size_x - text_w - 10, size_y - fontsize * 1.3),
                label,
                fill=text_fill,
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            # canvas.paste(img_crop, ((j+2)*size, i*size))
            canvas.paste(img_crop, (j * size_x, i * size_y))

    line_col = (192, 192, 192)
    line_draw = ImageDraw.Draw(canvas)

    # vertical
    nx = len(versions[0])
    line_draw.line((1.5, 0, 1.5, h), fill=line_col, width=3)
    for i in range(1, nx):
        line_draw.line((i * size_x, 0, i * size_x, h), fill=line_col, width=3)
    line_draw.line((w - 1.5, 0, w - 1.5, h), fill=line_col, width=3)

    # horizontal
    ny = len(versions)
    line_draw.line((0, 1.5, w * size_x, 1.5), fill=line_col, width=3)
    for i in range(1, ny):
        line_draw.line((0, i * size_y, w * size_x, i * size_y), fill=line_col, width=3)
    line_draw.line(
        (0, ny * size_y - 1.5, w * size_x, ny * size_y - 1.5), fill=line_col, width=3
    )
    canvas.show()

    if filename is not None:
        name = filename
        print("saving to", name)
        canvas.save(name)


def generate_distortion_comparison_lvc_g(save_file: bool):
    ref_bases = [
        "002.png",
    ]

    crop_coords = [
        (680, 270),
    ]

    crop_sizes = [
        (700, 550),
    ]

    versions = [
        [
            (
                FSS_DIR,
                "run24_keep",
                "",
                "orig_lvc_",
                "LCT",
            ),
            (
                FSS_DIR,
                "run24_keep",
                "",
                "orig_lvc_g_",
                "CV-Cast",
            ),
        ],
    ]

    name = "experiments/plots/dist_comparison.png" if save_file else None

    generate_image(
        name,
        versions,
        ref_bases,
        crop_coords,
        crop_sizes,
        filter=Image.Resampling.NEAREST,
        font_ratio=0.1,
    )


def generate_lvc_g(save_file: bool):
    ref_bases = [
        "003.png",
        "003.png",
        "003.png",
    ]

    crop_coords = [
        (486, 370),
        (486, 370),
        (486, 370),
    ]

    crop_sizes = [
        (200, 120),
        (200, 120),
        (200, 120),
    ]

    versions = [
        [
            (
                FSS_DIR,
                "run27_keep",
                "",
                "orig_ref_",
                "no distortion",
            ),
            (
                FSS_DIR,
                "run27_keep",
                "",
                "colorized_ref_",
                "no distortion",
            ),
        ],
        [
            (
                FSS_DIR,
                "run27_keep",
                "",
                "orig_lvc_",
                "LCT",
            ),
            (
                FSS_DIR,
                "run27_keep",
                "",
                "colorized_lvc_",
                "LCT",
            ),
        ],
        [
            (
                FSS_DIR,
                "run27_keep",
                "",
                "orig_lvc_g_",
                "CV-Cast",
            ),
            (
                FSS_DIR,
                "run27_keep",
                "",
                "colorized_lvc_g_",
                "CV-Cast",
            ),
        ],
    ]

    name = "experiments/plots/lvc_g_comparison.png" if save_file else None

    generate_image(
        name,
        versions,
        ref_bases,
        crop_coords,
        crop_sizes,
        filter=Image.Resampling.NEAREST,
        font_ratio=0.15,
    )


if __name__ == "__main__":
    # distortion comparison of optimized LVC
    generate_distortion_comparison_lvc_g(True)

    # optimized LVC comparison
    generate_lvc_g(True)
