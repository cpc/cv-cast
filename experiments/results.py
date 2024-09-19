import math
from pathlib import Path
from typing import List, Tuple, Literal

import numpy as np
import pandas as pd
import torch

from utils.q_search import nmapped_to_cr

MODEL_ORDER = {
    "fastseg_small": 0,
    "fastseg_large": 1,
    "yolov8_n": 2,
    "yolov8_s": 3,
    "yolov8_l": 4,
}

MODEL_UNITS = {
    "fastseg_small": ["mean_iu", "val_loss_avg"],
    "fastseg_large": ["mean_iu", "val_loss_avg"],
    "yolov8_n": ["mAP_50_95"],
    "yolov8_s": ["mAP_50_95"],
    "yolov8_l": ["mAP_50_95"],
}

DEFAULT_ID_COLS = [
    "probe_model_id",
    "model_id",
    "estimator",
    "mode",
    "nchunks",
    "csnr_db",
    "cr",
    "block_dct",
    "grad_w",
    "grad_sel",
    "grad_alloc",
]

OUTDIRS = {
    "disparity": [Path("experiments_tupu/runs/run54_keep")],
    "acc_vs_csnr": [
        Path("experiments_tupu/runs/run25_keep"),
        Path("experiments_tupu/runs/run26_keep"),
        Path("experiments_tupu/runs/run49_keep"),
        Path("experiments_tupu/runs/run50_keep"),
        Path("experiments_tupu/runs/run51_keep"),
    ],
    "acc_vs_csnr_sionna": [
        Path("experiments_tupu/runs/run03_keep"),
        Path("experiments_tupu/runs/run04_keep"),
        Path("experiments_tupu/runs/run05_keep"),
        Path("experiments_tupu/runs/run06_keep"),
        Path("experiments_tupu/runs/run07_keep"),
        Path("experiments_tupu/runs/run08_keep"),
        Path("experiments_tupu/runs/run13_keep"),
        Path("experiments_tupu/runs/run17_keep"),
        Path("experiments_tupu/runs/run20_keep"),
        Path("experiments_tupu/runs/run42_keep"),
        Path("experiments_tupu/runs/run43_keep"),
        Path("experiments_tupu/runs/run52_keep"),
        Path("experiments_tupu/runs/run53_keep"),
    ],
    "acc_vs_csnr_sionna_extra": [
        Path("experiments_tupu/runs/run03_keep"),
        Path("experiments_tupu/runs/run04_keep"),
        Path("experiments_tupu/runs/run05_keep"),
        Path("experiments_tupu/runs/run06_keep"),
        Path("experiments_tupu/runs/run07_keep"),
        Path("experiments_tupu/runs/run08_keep"),
        Path("experiments_tupu/runs/run13_keep"),
        Path("experiments_tupu/runs/run17_keep"),
        Path("experiments_tupu/runs/run20_keep"),
        Path("experiments_tupu/runs/run42_keep"),
        Path("experiments_tupu/runs/run43_keep"),
        Path("experiments_tupu/runs/run52_keep"),
        Path("experiments_tupu/runs/run53_keep"),
        Path("experiments_tupu/runs/run96_keep"),
        Path("experiments_tupu/runs/run97_keep"),
        Path("experiments_tupu/runs/run101_keep"),
        Path("experiments_tupu/runs/run102_keep"),
        Path("experiments_tupu/runs/run103_keep"),
        Path("experiments_tupu/runs/run109_keep"),
        Path("experiments_tupu/runs/run110_keep"),
        Path("experiments_tupu/runs/run111_keep"),
        Path("experiments_tupu/runs/run112_keep"),
        Path("experiments_tupu/runs/run113_keep"),
    ],
    "jpeg_grace": [
        Path("experiments_tupu/runs/run44_keep"),
        Path("experiments_tupu/runs/run45_keep"),
        Path("experiments_tupu/runs/run46_keep"),
        Path("experiments_tupu/runs/run47_keep"),
        Path("experiments_tupu/runs/run48_keep"),
    ],
    "jpeg_grace_sionna": [
        Path("experiments_tupu/runs/run32_keep"),
        Path("experiments_tupu/runs/run33_keep"),
        Path("experiments_tupu/runs/run34_keep"),
        Path("experiments_tupu/runs/run35_keep"),
        Path("experiments_tupu/runs/run36_keep"),
        Path("experiments_tupu/runs/run37_keep"),
        Path("experiments_tupu/runs/run38_keep"),
        Path("experiments_tupu/runs/run39_keep"),
        Path("experiments_tupu/runs/run40_keep"),
        Path("experiments_tupu/runs/run41_keep"),
    ],
    "tcm_sionna_8imgs_cr": [
        Path("experiments_tupu/runs/run61_keep"),
        Path("experiments_tupu/runs/run62_keep"),
        Path("experiments_tupu/runs/run63_keep"),
        Path("experiments_tupu/runs/run64_keep"),
        Path("experiments_tupu/runs/run65_keep"),
        Path("experiments_tupu/runs/run66_keep"),
        Path("experiments_tupu/runs/run67_keep"),
        Path("experiments_tupu/runs/run68_keep"),
        Path("experiments_tupu/runs/run69_keep"),
        Path("experiments_tupu/runs/run70_keep"),
    ],
    "tcm_sionna": [
        Path("experiments_tupu/runs/run73_keep"),
        Path("experiments_tupu/runs/run74_keep"),
        Path("experiments_tupu/runs/run75_keep"),
        Path("experiments_tupu/runs/run76_keep"),
        Path("experiments_tupu/runs/run77_keep"),
        Path("experiments_tupu/runs/run78_keep"),
        Path("experiments_tupu/runs/run83_keep"),
        Path("experiments_tupu/runs/run84_keep"),
        Path("experiments_tupu/runs/run85_keep"),
        Path("experiments_tupu/runs/run89_keep"),
        Path("experiments_tupu/runs/run90_keep"),
        Path("experiments_tupu/runs/run91_keep"),
        Path("experiments_tupu/runs/run92_keep"),
        Path("experiments_tupu/runs/run93_keep"),
        Path("experiments_tupu/runs/run94_keep"),
        Path("experiments_tupu/runs/run95_keep"),
        Path("experiments_tupu/runs/run98_keep"),
        Path("experiments_tupu/runs/run99_keep"),
        Path("experiments_tupu/runs/run100_keep"),
    ],
    "acc_vs_csnr_precise": [Path("experiments_hupu/runs/run05_keep")],
}

PROBE_DIR_OLD = Path("experiments_hupu/runs/run14_keep")
PROBE_DIR = Path("experiments_tupu/runs/run24_keep")

_DF_CACHE = {}


def collect_results(
    outdirs: List[Path], partial: bool = True
) -> Tuple[List[dict], List[list]]:
    """Collects results gathered by get_accuracies() in the case of error"""

    res_probe = []
    res_lvc = []

    for outdir in outdirs:
        probe_file = outdir / "probe_results.pt"
        lvc_file = outdir / "lvc_results.pt"

        if partial:
            tmp_res = {}

            for p in (outdir / "partial").glob("probe_models_*.pt"):
                r = torch.load(p)
                tmp_res.update(r)

            res_probe.append(tmp_res)

            if probe_file.exists():
                probe_file.rename(probe_file.with_stem(probe_file.stem + "_old"))

            torch.save(tmp_res, probe_file)
        else:
            try:
                res_probe = torch.load(probe_file)
            except FileNotFoundError:
                res_probe = torch.load(probe_file.with_stem(probe_file.stem + "_old"))

        if partial:
            tmp_res = []

            for p in (outdir / "partial").glob("run_models_*.pt"):
                r = torch.load(p)
                tmp_res.append(r)

            res_lvc.append(tmp_res)

            if lvc_file.exists():
                lvc_file.rename(lvc_file.with_stem(lvc_file.stem + "_old"))

            torch.save(res_lvc, lvc_file)
        else:
            try:
                res_lvc = torch.load(lvc_file)
            except FileNotFoundError:
                res_lvc = torch.load(lvc_file.with_stem(lvc_file.stem + "_old"))

    return (res_probe, res_lvc)


def get_result_rows(lvc_results_list, outdirs):
    rows = []
    for lvc_results, outdir in zip(lvc_results_list, outdirs):
        for lvc_result in lvc_results:
            model_id = lvc_result["model_id"]
            try:
                probe_model_id = lvc_result["probe_model_id"]
            except KeyError:
                probe_model_id = model_id
            units = MODEL_UNITS[model_id]
            scores = {}
            for i, unit in enumerate(units):
                scores[f"score{i}_lvc"] = lvc_result["res_lvc"].get(unit)
                scores[f"score{i}_lvc_g"] = lvc_result["res_lvc_g"].get(unit)

                try:
                    scores[f"score{i}_lvc_reprobe"] = lvc_result["res_lvc_reprobe"].get(
                        unit
                    )
                    scores[f"score{i}_lvc_g_reprobe"] = lvc_result[
                        "res_lvc_g_reprobe"
                    ].get(unit)
                except KeyError:
                    pass

                scores[f"unit{i}"] = unit

            try:
                grad_sel = lvc_result["grad_tasks"]["grad_norm_select"]
            except KeyError:
                grad_sel = lvc_result["grad_tasks"]["g_select"]

            try:
                grad_alloc = lvc_result["grad_tasks"]["grad_norm_allocate"]
            except KeyError:
                grad_alloc = lvc_result["grad_tasks"]["g_allocate"]

            try:
                grad_yuv = lvc_result["grad_tasks"]["w"]
            except KeyError:
                grad_yuv = lvc_result["grad_tasks"]["g_yuv"]

            try:
                jpeg_config = lvc_result["jpeg_config"]
            except KeyError:
                jpeg_config = None

            try:
                sionna_config = lvc_result["sionna_config"]
            except KeyError:
                sionna_config = None

            d = {
                "probe_model_id": probe_model_id,
                "model_id": model_id,
                "estimator": lvc_result["lvc_params"]["estimator"],
                "mode": lvc_result["lvc_params"]["mode"],
                "nchunks": lvc_result["lvc_params"]["nchunks"],
                "csnr_db": lvc_result["lvc_params"]["csnr_db"],
                "cr": lvc_result["lvc_params"]["cr"],
                "block_dct": all(
                    [
                        x in lvc_result["lvc_params"]
                        for x in ["dct_w", "dct_h", "grouping"]
                    ]
                ),
                "grad_w": grad_yuv,
                "grad_sel": grad_sel,
                "grad_alloc": grad_alloc,
                "outdir": outdir,
                **scores,
            }

            if jpeg_config is not None:
                d["codec"] = jpeg_config["codec"]
                d["param"] = jpeg_config["param"]

            if sionna_config is not None:
                d["nbits_per_sym"] = sionna_config["nbits_per_sym"]
                try:
                    d["coderate"] = sionna_config["coderate"]
                except KeyError:
                    if jpeg_config is None:
                        coderate = 1.0
                    else:
                        coderate = 0.5
                    d["coderate"] = coderate

            if jpeg_config is not None and sionna_config is not None:
                try:
                    eq_crs = []
                    enc_sizes = []

                    if d["codec"] in ("jpeg", "icm", "tcm"):
                        key = "res_jpeg"
                    elif d["codec"] == "grace":
                        key = "res_jpeg_g"

                    for jpeg_res in lvc_result[key]:
                        img_w = jpeg_res["img_w"]
                        img_h = jpeg_res["img_h"]
                        enc_size = jpeg_res["enc_size"]
                        nsymbols = math.ceil(
                            enc_size * 8 / d["coderate"] / d["nbits_per_sym"]
                        )
                        eq_crs.append(
                            nmapped_to_cr(nsymbols, img_w, img_h, mode=d["mode"])
                        )
                        enc_sizes.append(enc_size)

                    d["eq_cr"] = np.array(eq_crs).mean()
                    d["enc_size"] = np.array(enc_sizes).mean()
                except KeyError:
                    pass

            rows.append(d)
    return rows


def collect_df(
    df_name: Literal[
        "disparity",
        "acc_vs_csnr",
        "acc_vs_csnr_sionna",
        "acc_vs_csnr_sionna_extra",
        "jpeg_grace",
        "jpeg_grace_sionna",
        "tcm_sionna",
        "tcm_sionna_8imgs_cr",
        "acc_vs_csnr_precise",
    ],
    do_reload: bool = False,
    partial: bool = True,
) -> Tuple[List[dict], pd.DataFrame]:
    global _DF_CACHE
    outdirs = OUTDIRS[df_name]

    if not df_name in _DF_CACHE:
        _DF_CACHE[df_name] = None

    if do_reload:
        _DF_CACHE[df_name] = None

    if _DF_CACHE[df_name] is None:
        probe_results_list, lvc_results_list = collect_results(outdirs, partial=partial)
        rows_disparity = get_result_rows(lvc_results_list, outdirs)
        _DF_CACHE[df_name] = (probe_results_list, pd.DataFrame(rows_disparity))

    return _DF_CACHE[df_name]


def get_baseline(probe_results_list: List[dict]) -> Tuple[dict, dict]:
    baseline0 = {}
    baseline1 = {}

    for probe_results in probe_results_list:
        for model_id, res in probe_results.items():
            for unit, score in res["orig"].items():
                if len(MODEL_UNITS[model_id]) > 0:
                    if unit == MODEL_UNITS[model_id][0]:
                        baseline0[model_id] = score

                if len(MODEL_UNITS[model_id]) > 1:
                    if unit == MODEL_UNITS[model_id][1]:
                        baseline1[model_id] = score

    return (baseline0, baseline1)
