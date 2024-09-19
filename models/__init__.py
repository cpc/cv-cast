import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Literal, TypedDict

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from lvc.lvc import GradConfig, JPEGConfig, create_lvc, SionnaConfig, fmt_sionna
from utils import set_device
from utils.q_search import fetch_param, DEFAULT_FILTERS, QSearchFilters
from .model import Model, ModelConfig
from .fastseg import FastsegModel
from .yolov8 import YoloModel
from .drn import DrnModel


class JPEGInitParam(TypedDict):
    """Input parameters for JPEG encoding"""

    codec: Literal["jpeg", "grace", "icm", "tcm"]
    turbojpeg_enc: bool
    turbojpeg_dec: bool
    forced_param: int | float | str | None


def get_model(
    config: ModelConfig,
    device: str | torch.device | None = None,
    num_batches: int | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    lvc_chain: transforms.Compose | None = None,
    color_space: str = "rgb",
    do_print: bool = False,
) -> Model:
    name = config["name"]

    if name == "yolov8":
        return YoloModel(
            config,
            device=device,
            num_batches=num_batches,
            batch_size=batch_size,
            num_workers=num_workers,
            lvc_chain=lvc_chain,
            color_space=color_space,
            do_print=do_print,
        )
    elif name == "fastseg":
        return FastsegModel(
            config,
            device=device,
            num_batches=num_batches,
            batch_size=batch_size,
            num_workers=num_workers,
            lvc_chain=lvc_chain,
            color_space=color_space,
            do_print=do_print,
        )
    elif name == "drn":
        return DrnModel(
            config,
            device=device,
            num_batches=num_batches,
            batch_size=batch_size,
            num_workers=num_workers,
            lvc_chain=lvc_chain,
            color_space=color_space,
            do_print=do_print,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def get_model_id(model_config: ModelConfig):
    return "_".join([model_config["name"], model_config["variant"]])


# def grad_config(task, w, gnorm) -> GradConfig:
#     return {
#         "w": w if task["w"] else None,
#         "grad_norm_select": gnorm if task["grad_norm_select"] else None,
#         "grad_norm_allocate": gnorm if task["grad_norm_allocate"] else None,
#     }


def probe_models(
    configs: List[ModelConfig],
    grad_type: Literal["dist_sq", "dist_abs"],
    possible_nchunks: List[int],
    device: torch.device | str,
    color_space: str,
    rank: int,
    cpus: np.ndarray | None = None,
    num_workers: int = 0,
    num_batches: int | None = None,
    batch_size: int = 1,
    num_batches_probe: int | None = 32,
    batch_size_probe: int = 1,
    dct_size: Tuple[int, int] | None = None,
    subtract_mean: bool = True,
    scale: float = 1.0,
    norm: Literal["abs", "sq"] = "sq",
    do_print: bool = False,
    outdir: Path | None = None,
    dry: bool = False,
) -> dict:
    if cpus is not None:
        time.sleep(rank * 0.01)
        os.sched_setaffinity(0, set(cpus.tolist()))
        affinity = os.sched_getaffinity(0)
        print(f"Rank {rank} running on CPUs: {affinity}")
        torch.set_num_threads(len(cpus))

    device = set_device(device, do_print=do_print)

    res = {}
    if len(configs) == 0:
        return res

    # Early detect duplicate entries
    for model_config in configs:
        model_id = get_model_id(model_config)

        if res.get(model_id) is None:
            res[model_id] = {}
        else:
            raise ValueError(f"Duplicate model entry '{model_id}'")

    for i, model_config in enumerate(configs):
        start_time = time.perf_counter()
        model_id = get_model_id(model_config)
        msg = (
            f"({i:3d}/{len(configs):3d}) rank {rank:2d}, model {model_id}, eval + probe"
        )
        print(f"-- {msg} (start) --")

        # Eval original
        model_eval = get_model(
            model_config,
            device=device,
            num_batches=num_batches,
            batch_size=batch_size,
            num_workers=num_workers,
            color_space=color_space,
            do_print=do_print,
        )

        if not dry:
            result_orig = model_eval.eval()
        else:
            result_orig = {}

        # Extract gradients
        model_probe = get_model(
            model_config,
            device=device,
            num_batches=num_batches_probe,
            batch_size=batch_size_probe,
            num_workers=num_workers,
            color_space=color_space,
            do_print=do_print,
        )

        if not dry:
            probe_res = model_probe.run_probe(
                grad_type,
                possible_nchunks,
                dct_size=dct_size,
                subtract_mean=subtract_mean,
                scale=scale,
                norm=norm,
            )
            partial_res = {
                "orig": result_orig,
                "dct_var": probe_res["dct_var"],
                "grads_rgb": probe_res["grads_rgb"],
                "grads_yuv": probe_res["grads_yuv"],
                "grads_yuv_420": probe_res["grads_yuv_420"],
                "grads_norm": probe_res["grads_norm"],
                "grads_norm_420": probe_res["grads_norm_420"],
                "w_yuv": probe_res["W"],
            }
        else:
            partial_res = {}

        res[model_id] = partial_res

        duration = int(time.perf_counter() - start_time)
        duration_msg = f"{duration // 60:02d}m{duration % 60:02d}s"

        if outdir is None:
            print(f"== {msg} {duration_msg} (end) ==")
        else:
            fname_base = f"probe_models_rank{rank:02d}_{i:03d}.pt"
            fname = outdir / fname_base
            print(
                f"== {msg} {duration_msg} (end) -> {'/'.join(outdir.parts[-2:])}/{fname_base} =="
            )
            if not dry:
                torch.save({model_id: partial_res}, fname)

    return res


def run_models(
    configs: List[
        Tuple[
            dict,
            dict,
            ModelConfig | Tuple[ModelConfig, ModelConfig],
            JPEGInitParam | None,
            SionnaConfig | None,
        ]
    ],
    probe_results: dict,  # result from probe_models()
    device: torch.device | str,
    color_space: str,
    rank: int,
    cpus: np.ndarray | None = None,
    num_workers: int = 0,
    num_batches: int | None = None,
    batch_size: int = 16,
    num_batches_probe: int | None = 16,
    batch_size_probe: int = 4,
    do_print: bool = False,
    outdir: Path | None = None,
    dry: bool = False,
    reprobe: bool = False,
    reprobe_g: bool = False,
    g_opt: bool = True,  # run gradient-driven optimizations
    grad_type: Literal["dist_sq", "dist_abs"] = "dist_sq",  # used for reprobing
    q_search_files: List[str | Path] = [],
    q_search_filters: QSearchFilters = DEFAULT_FILTERS,
) -> List[dict]:
    # Deprecate reprobing: model.run_probe below would need additional params passed
    assert reprobe == False
    assert reprobe_g == False

    if cpus is not None:
        time.sleep(rank * 0.01)
        os.sched_setaffinity(0, set(cpus.tolist()))
        affinity = os.sched_getaffinity(0)
        print(f"Rank {rank} running on CPUs: {affinity}")
        torch.set_num_threads(len(cpus))

    device = set_device(device, do_print=do_print)

    try:
        q_search_results = pd.concat(
            [pd.DataFrame(torch.load(f)) for f in q_search_files], ignore_index=True
        )
    except ValueError:
        q_search_results = pd.DataFrame()

    res = []
    if len(configs) == 0:
        return res

    for i, (
        lvc_param_dict,
        grad_task,
        model_conf,
        jpeg_init_param,
        sionna_config,
    ) in enumerate(configs):
        start_time = time.perf_counter()

        if isinstance(model_conf, tuple):
            probe_model_config, model_config = model_conf
            probe_model_id = get_model_id(probe_model_config)
            model_id = get_model_id(model_config)
        else:
            model_config = model_conf
            model_id = get_model_id(model_config)
            probe_model_id = model_id

        cr = lvc_param_dict["cr"]
        csnr_db = str(lvc_param_dict["csnr_db"])
        nchunks = lvc_param_dict["nchunks"]
        est = lvc_param_dict["estimator"]
        mode = lvc_param_dict["mode"]
        blk = "ff" if lvc_param_dict.get("dct_w") is None else "bb"

        task_msg = ":".join(
            [
                f"{n if v else ''}"
                for n, v in zip(["", "gyuv", "gsel", "gpow"], grad_task.values())
            ]
        )

        param_name = {
            "jpeg": "Q",
            "grace": "B",
            "icm": "model",
            "tcm": "model",
        }

        param_fmt = {
            "jpeg": "{:12}",
            "grace": "{:12.5e}",
            "icm": " {}",
            "tcm": " {}",
        }

        if jpeg_init_param is None:
            jpeg_config = None
        elif jpeg_init_param["codec"] in ("jpeg", "grace"):
            if sionna_config is None:
                raise ValueError("JPEG params require Sionna config")

            if jpeg_init_param["forced_param"] is not None:
                raise ValueError("JPEG/GRACE does not support forced param")

            param = fetch_param(
                q_search_results,
                model_id,
                jpeg_init_param["codec"],
                sionna_config["nbits_per_sym"],
                cr,
                q_search_filters,
            )

            if param is None:
                raise ValueError(
                    f"Could not find param for {jpeg_init_param}, nbits {sionna_config['nbits_per_sym']}, CR: {cr} in \n{q_search_results}"
                )

            jpeg_config = {
                "codec": jpeg_init_param["codec"],
                "param": param,
                "param_name": param_name[jpeg_init_param["codec"]],
                "param_fmt": param_fmt[jpeg_init_param["codec"]],
                "turbojpeg_enc": jpeg_init_param["turbojpeg_enc"],
                "turbojpeg_dec": jpeg_init_param["turbojpeg_dec"],
            }
        elif jpeg_init_param["codec"] in ("icm", "tcm"):
            if jpeg_init_param["forced_param"] is None:
                raise ValueError(
                    "Non-JPEG/GRACE codec does not support param search (must use forced_param)"
                )

            jpeg_config = {
                "codec": jpeg_init_param["codec"],
                "param": jpeg_init_param["forced_param"],
                "param_name": param_name[jpeg_init_param["codec"]],
                "param_fmt": param_fmt[jpeg_init_param["codec"]],
                "turbojpeg_enc": jpeg_init_param["turbojpeg_enc"],
                "turbojpeg_dec": jpeg_init_param["turbojpeg_dec"],
            }
        else:
            raise ValueError(f"Unknown codec {jpeg_init_param['codec']}")

        if jpeg_config is None:
            jpeg_msg = "jpeg    no"
        else:
            jpeg_msg = f"{jpeg_config['codec']:5s} {jpeg_config['param_name']}{jpeg_config['param_fmt'].format(jpeg_config['param'])}"

        if sionna_config is None:
            snmsg = "sionna           no"
        else:
            snmsg = f"sionna {fmt_sionna(sionna_config)}"

        msg = (
            f"({i:3d}/{len(configs):3d}) r{rank:d},p|{probe_model_id:14s},e|{model_id:14s},"
            + f"{est:4s},{mode:3d},{nchunks:4d},{csnr_db:3s}dB,CR {cr:.4f},{blk:2s},{task_msg:11s},{jpeg_msg:9s},{snmsg}"
        )
        print(f"-- {msg}         (start) --")

        grad_key = "grads_norm_420" if mode == 420 else "grads_norm"
        grad_yuv_key = "grads_yuv_420" if mode == 420 else "grads_yuv"

        if not dry:
            probe_result = probe_results[probe_model_id]
            grads_norm = probe_result[grad_key][nchunks]
            grads_mean = probe_result[grad_yuv_key]
            w_yuv = probe_result["w_yuv"].to(device, non_blocking=True)
        else:
            grads_norm = torch.zeros(
                3, int(math.sqrt(nchunks)), int(math.sqrt(nchunks))
            )
            grads_mean = torch.zeros(
                3, int(math.sqrt(nchunks)), int(math.sqrt(nchunks))
            )
            w_yuv = torch.zeros(3)

        results = {"noise": None, "jpeg": []}

        # Eval with original LVC
        if do_print:
            print("\nLVC ORIGINAL:")

        if jpeg_config is None or jpeg_config["codec"] != "grace":
            # GRACE requires gradients
            lvc_chain = create_lvc(
                lvc_param_dict,
                device,
                False,
                unsqueeze=True,
                do_print=False,
                results=results,
                grad_config=None,
                sionna_config=sionna_config,
                jpeg_config=jpeg_config,
            )

            model_lvc = get_model(
                model_config,
                device=device,
                num_batches=num_batches,
                batch_size=batch_size,
                num_workers=num_workers,
                lvc_chain=lvc_chain,
                color_space=color_space,
                do_print=do_print,
            )

            if not dry:
                res_lvc = model_lvc.eval()
            else:
                res_lvc = {model_config["unit"]: -1.0}
        else:
            results = {"noise": None, "jpeg": None}
            res_lvc = {model_config["unit"]: -1.0}

        if g_opt and (
            jpeg_config is None or jpeg_config["codec"] not in ("jpeg", "icm", "tcm")
        ):  # plain JPEG is not grad-optimized
            # Eval with gradient-optimized LVC
            if do_print:
                print("\nLVC GRADIENTS:")

            grad_config: GradConfig = {
                "type": grad_task["type"],
                "g_yuv": grad_task["g_yuv"],
                "g_select": grad_task["g_select"],
                "g_allocate": grad_task["g_allocate"],
                "w": w_yuv,
                "grad_mean": grads_mean,
                "grad_norm": grads_norm,
            }

            results_g = {"noise": None, "jpeg": []}

            lvc_chain_g = create_lvc(
                lvc_param_dict,
                device,
                False,
                unsqueeze=True,
                do_print=False,
                results=results_g,
                grad_config=grad_config,
                sionna_config=sionna_config,
                jpeg_config=jpeg_config,
            )

            model_lvc_g = get_model(
                model_config,
                device=device,
                num_batches=num_batches,
                batch_size=batch_size,
                num_workers=num_workers,
                lvc_chain=lvc_chain_g,
                color_space=color_space,
                do_print=do_print,
            )

            if not dry:
                res_lvc_g = model_lvc_g.eval()
            else:
                res_lvc_g = {model_config["unit"]: -1.0}
                results_g = {"noise": None, "jpeg": None}

            if jpeg_config is None or jpeg_config["codec"] != "grace":
                # Re-probe with original LVC and eval again
                if do_print:
                    print("\nLVC GRADIENTS (reprobe orig):")

                model_probe_lvc = get_model(
                    model_config,
                    device=device,
                    num_batches=num_batches_probe,
                    batch_size=batch_size_probe,
                    num_workers=num_workers,
                    lvc_chain=lvc_chain,
                    color_space=color_space,
                    do_print=do_print,
                )

                if not dry and reprobe:
                    probe_res_lvc = model_probe_lvc.run_probe(grad_type, [nchunks])
                    grads_norm_lvc = probe_res_lvc[grad_key][nchunks]
                    grads_mean_lvc = probe_res_lvc[grad_yuv_key]
                    w_yuv_lvc = probe_res_lvc["W"]
                else:
                    grads_mean_lvc = torch.zeros(
                        3, model_probe_lvc.img_h, model_probe_lvc.img_w
                    )
                    grads_norm_lvc = torch.zeros(
                        3, int(math.sqrt(nchunks)), int(math.sqrt(nchunks))
                    )
                    w_yuv_lvc = torch.zeros(3)

                grad_config: GradConfig = {
                    "type": grad_task["type"],
                    "g_yuv": grad_task["g_yuv"],
                    "g_select": grad_task["g_select"],
                    "g_allocate": grad_task["g_allocate"],
                    "w": w_yuv_lvc,
                    "grad_mean": grads_mean_lvc,
                    "grad_norm": grads_norm_lvc,
                }

                results_reprobe = {"noise": None, "jpeg": []}

                if not dry and reprobe:
                    lvc_chain_reprobe = create_lvc(
                        lvc_param_dict,
                        device,
                        False,
                        unsqueeze=True,
                        do_print=False,
                        results=results_reprobe,
                        grad_config=grad_config,
                        sionna_config=sionna_config,
                        jpeg_config=jpeg_config,
                    )

                    model_lvc_reprobe = get_model(
                        model_config,
                        device=device,
                        num_batches=num_batches,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        lvc_chain=lvc_chain_reprobe,
                        color_space=color_space,
                        do_print=do_print,
                    )

                    res_lvc_reprobe = model_lvc_reprobe.eval()
                else:
                    res_lvc_reprobe = {}
            else:
                results_reprobe = {"noise": None, "jpeg": None}
                res_lvc_reprobe = {}
                grads_mean_lvc = torch.zeros(3, model_lvc_g.img_h, model_lvc_g.img_w)
                grads_norm_lvc = torch.zeros(
                    3, int(math.sqrt(nchunks)), int(math.sqrt(nchunks))
                )
                w_yuv_lvc = torch.zeros(3)

            res_lvc_reprobe.update(
                {
                    "w_yuv": w_yuv_lvc,
                    "grads_norm": grads_norm_lvc,
                }
            )

            # Re-probe with gradient-optimized LVC and eval again
            if do_print:
                print("\nLVC GRADIENTS (reprobe grad):")

            model_probe_lvc_g = get_model(
                model_config,
                device=device,
                num_batches=num_batches_probe,
                batch_size=batch_size_probe,
                num_workers=num_workers,
                lvc_chain=lvc_chain_g,
                color_space=color_space,
                do_print=do_print,
            )

            if not dry and reprobe_g:
                probe_res_lvc_g = model_probe_lvc_g.run_probe(grad_type, [nchunks])
                grads_norm_lvc_g = probe_res_lvc_g[grad_key][nchunks]
                grads_mean_lvc_g = probe_res_lvc_g[grad_yuv_key]
                w_yuv_lvc_g = probe_res_lvc_g["W"]
            else:
                grads_mean_lvc_g = torch.zeros(
                    3, model_probe_lvc_g.img_h, model_probe_lvc_g.img_w
                )
                grads_norm_lvc_g = torch.zeros(
                    3, int(math.sqrt(nchunks)), int(math.sqrt(nchunks))
                )
                w_yuv_lvc_g = torch.zeros(3)

            grad_config: GradConfig = {
                "type": grad_task["type"],
                "g_yuv": grad_task["g_yuv"],
                "g_select": grad_task["g_select"],
                "g_allocate": grad_task["g_allocate"],
                "w": w_yuv_lvc_g,
                "grad_mean": grads_mean_lvc_g,
                "grad_norm": grads_norm_lvc_g,
            }

            results_g_reprobe = {"noise": None, "jpeg": []}

            if not dry and reprobe_g:
                lvc_chain_g_reprobe = create_lvc(
                    lvc_param_dict,
                    device,
                    False,
                    unsqueeze=True,
                    do_print=False,
                    results=results_g_reprobe,
                    grad_config=grad_config,
                    sionna_config=sionna_config,
                    jpeg_config=jpeg_config,
                )

                model_lvc_g_reprobe = get_model(
                    model_config,
                    device=device,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    lvc_chain=lvc_chain_g_reprobe,
                    color_space=color_space,
                    do_print=do_print,
                )

                res_lvc_g_reprobe = model_lvc_g_reprobe.eval()
            else:
                res_lvc_g_reprobe = {}

            res_lvc_g_reprobe.update(
                {
                    "w_yuv": w_yuv_lvc_g,
                    "grads_norm": grads_norm_lvc_g,
                }
            )
        else:
            results_g = {"noise": None, "jpeg": None}
            results_g_reprobe = {"noise": None, "jpeg": None}
            res_lvc_g = {model_config["unit"]: -1.0}
            res_lvc_reprobe = {}
            res_lvc_g_reprobe = {}

        partial_res = {
            "rank": rank,
            "probe_model_id": probe_model_id,
            "model_id": model_id,
            "lvc_params": lvc_param_dict,
            "color_space": color_space,
            "grads_norm": grads_norm,
            "grad_tasks": grad_task,
            "jpeg_config": jpeg_config,
            "sionna_config": sionna_config,
            "res_lvc": res_lvc,
            "res_lvc_g": res_lvc_g,
            # "res_lvc_reprobe": res_lvc_reprobe,
            # "res_lvc_g_reprobe": res_lvc_g_reprobe,
            "res_jpeg": results["jpeg"],
            "res_jpeg_g": results_g["jpeg"],
            # "res_jpeg_reprobe": results_reprobe["jpeg"],
            # "res_jpeg_g_reprobe": results_g_reprobe["jpeg"],
        }

        acc_out = res_lvc_g[model_config["unit"]]

        if jpeg_config is not None and jpeg_config["codec"] in ("jpeg", "icm", "tcm"):
            acc_out = res_lvc[model_config["unit"]]

        res.append(partial_res)

        duration = int(time.perf_counter() - start_time)
        duration_msg = f"{duration // 60:03d}m{duration % 60:02d}s"

        if outdir is None:
            print(f"== {msg} {duration_msg} (end) | ACC {acc_out:5.3f} ==")
        else:
            fname_base = f"run_models_rank{rank:02d}_{i:03d}.pt"
            fname = outdir / fname_base
            print(
                f"== {msg} {duration_msg} (end) -> {'/'.join(outdir.parts[-2:])}/{fname_base} | ACC {acc_out:5.3f} =="
            )
            if not dry:
                torch.save(partial_res, fname)

    return res
