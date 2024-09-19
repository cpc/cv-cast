import itertools
import json
import math
import os
import pprint
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import torch

from lvc.lvc import create_lvc, GradConfig, SionnaConfig, JPEGConfig
from models import get_model, run_models, probe_models, get_model_id
from models.model import plot_channels, ModelConfig
from utils import block_process, set_device, Size
from utils.q_search import run_qsearch, fetch_param, DEFAULT_FILTERS, QSearchFilters

SEED = 42

MODEL_FILES = {
    "icm": "/media/data2/jakub/data/models/ICM-v1/icm.pth.tar",
    "tcm_n64_lambda0.0025": "/media/data2/jakub/data/models/LIC_TCM/0.0025.pth.tar",
    "tcm_n64_lambda0.0035": "/media/data2/jakub/data/models/LIC_TCM/0.0035.pth.tar",
    "tcm_n64_lambda0.0067": "/media/data2/jakub/data/models/LIC_TCM/0.0067.pth.tar",
    "tcm_n64_lambda0.013": "/media/data2/jakub/data/models/LIC_TCM/0.013.pth.tar",
    "tcm_n64_lambda0.025": "/media/data2/jakub/data/models/LIC_TCM/0.025.pth.tar",
    "tcm_n64_lambda0.05": "/media/data2/jakub/data/models/LIC_TCM/0.05.pth.tar",
    "tcm_n128_lambda0.05": "/media/data2/jakub/data/models/LIC_TCM/mse_lambda_0.05.pth.tar",
}


def gradient_variance(outdir: Path):
    """See how the selection of probe samples affects the results"""

    print("Gradient variances")
    device = set_device("cuda", do_print=True)
    color_space = "rgb"

    num_batches = [1, 4, 16, 32, 64]
    batch_sizes = [1, 2, 4]
    seeds = range(10)

    config = {
        "name": "fastseg",
        "variant": "small",
        "snapshot": (
            Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
        ),
        "unit": "mean_iu",
    }

    for nbatches, batch_size in itertools.product(num_batches, batch_sizes):
        res: torch.Tensor | None = None

        for seed in seeds:
            torch.manual_seed(seed)

            model_probe = get_model(
                config,
                device,
                num_batches=nbatches,
                batch_size=batch_size,
                color_space=color_space,
            )

            result_probe = model_probe.run_probe("dist_sq", [])

            if res is None:
                res = result_probe[f"grads_{color_space}"].unsqueeze(0)
            else:
                res = torch.cat(
                    [res, result_probe[f"grads_{color_space}"].unsqueeze(0)]
                )

        if res is None:
            raise ValueError("No results were gathered")

        std = res.std(dim=0)

        plot_channels(
            std.cpu().numpy(),
            f"gradient std ({color_space.upper()})",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_grads_std_nb{nbatches}_bs{batch_size}.png"
            ),
        )


def gradient_maps(outdir: Path):
    """Generate gradient maps"""

    print("Gradient maps")
    torch.manual_seed(SEED)
    gpu_i = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_i}"
    device = set_device(f"cuda:0")
    cpus = set(range(84, 96))
    os.sched_setaffinity(0, cpus)
    affinity = os.sched_getaffinity(0)
    print(f"Running on CPUs: {affinity}")
    torch.set_num_threads(len(cpus))

    color_space = "yuv"
    modes = [444]  # , 420]
    num_batches_probe = 32
    batch_size_probe = 1

    configs = [
        {
            "name": "fastseg",
            "variant": "small",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "fastseg",
            "variant": "large",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/large/best_checkpoint_ep172.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "yolov8",
            "variant": "n",
            "snapshot": "yolov8n.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "s",
            "snapshot": "yolov8s.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "l",
            "snapshot": "yolov8l.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
    ]

    for config, mode in itertools.product(configs, modes):
        model_probe = get_model(
            config, device, num_batches_probe, batch_size_probe, color_space=color_space
        )

        result_probe = model_probe.run_probe("dist_sq", [64, 256, 1024])

        model_id = "_".join([config["name"], config["variant"]])
        fname = outdir / f"probe_result_full_{model_id}_{mode}.pt"
        torch.save(result_probe, fname)

        grad_key = "grads_norm_420" if mode == 420 else "grads_norm"
        grad_yuv_key = "grads_yuv_420" if mode == 420 else "grads_yuv"

        grads_norm_64 = result_probe[grad_key][64].numpy()
        grads_norm_256 = result_probe[grad_key][256].numpy()
        grads_norm_1024 = result_probe[grad_key][1024].numpy()
        grads_norm_sq_64 = result_probe[grad_key][64].square().numpy()
        grads_norm_sq_256 = result_probe[grad_key][256].square().numpy()
        grads_norm_sq_1024 = result_probe[grad_key][1024].square().numpy()
        dct_var_64 = result_probe["dct_var"][64].numpy()
        dct_var_256 = result_probe["dct_var"][256].numpy()
        dct_var_1024 = result_probe["dct_var"][1024].numpy()

        prod_64 = grads_norm_sq_64 * dct_var_64
        prod_256 = grads_norm_sq_256 * dct_var_256
        prod_1024 = grads_norm_sq_1024 * dct_var_1024

        plot_channels(
            grads_norm_64,
            f"block_norm DCT gradient ({color_space.upper()}, 64)",
            show=False,
            save=(
                outdir / f"{config['name']}{config['variant']}_grads_norm_{mode}_64.png"
            ),
            log=True,
        )

        plot_channels(
            grads_norm_256,
            f"block_norm DCT gradient ({color_space.upper()}, 256)",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_grads_norm_{mode}_256.png"
            ),
            log=True,
        )

        plot_channels(
            grads_norm_1024,
            f"block_norm DCT gradient ({color_space.upper()}, 1024)",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_grads_norm_{mode}_1024.png"
            ),
            log=True,
        )

        plot_channels(
            grads_norm_sq_64,
            f"block_norm sq DCT gradient ({color_space.upper()}, 64)",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_grads_norm_sq_{mode}_64.png"
            ),
            log=True,
        )

        plot_channels(
            grads_norm_sq_256,
            f"block_norm sq DCT gradient ({color_space.upper()}, 256)",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_grads_norm_sq_{mode}_256.png"
            ),
            log=True,
        )

        plot_channels(
            grads_norm_sq_1024,
            f"block_norm sq DCT gradient ({color_space.upper()}, 1024)",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_grads_norm_sq_{mode}_1024.png"
            ),
            log=True,
        )

        plot_channels(
            dct_var_64,
            f"DCT variance ({color_space.upper()}, 64)",
            show=False,
            save=(
                outdir / f"{config['name']}{config['variant']}_dct_var_{mode}_64.png"
            ),
            log=True,
        )

        plot_channels(
            dct_var_256,
            f"DCT variance ({color_space.upper()}, 256)",
            show=False,
            save=(
                outdir / f"{config['name']}{config['variant']}_dct_var_{mode}_256.png"
            ),
            log=True,
        )

        plot_channels(
            dct_var_1024,
            f"DCT variance ({color_space.upper()}, 1024)",
            show=False,
            save=(
                outdir / f"{config['name']}{config['variant']}_dct_var_{mode}_1024.png"
            ),
            log=True,
        )

        plot_channels(
            prod_64,
            f"Variance * squared grad ({color_space.upper()}, 64)",
            show=False,
            save=(outdir / f"{config['name']}{config['variant']}_prod_{mode}_64.png"),
            log=True,
        )

        plot_channels(
            prod_256,
            f"Variance * squared grad ({color_space.upper()}, 256)",
            show=False,
            save=(outdir / f"{config['name']}{config['variant']}_prod_{mode}_256.png"),
            log=True,
        )

        plot_channels(
            prod_1024,
            f"Variance * squared grad ({color_space.upper()}, 1024)",
            show=False,
            save=(outdir / f"{config['name']}{config['variant']}_prod_{mode}_1024.png"),
            log=True,
        )

        plot_channels(
            result_probe[f"grads_{color_space}"].abs().cpu().numpy(),
            f"abs DCT gradient ({color_space.upper()}))",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_grads_{color_space}_abs_{mode}.png"
            ),
            log=True,
        )

        plot_channels(
            result_probe[f"dct_{color_space}"].abs().cpu().numpy(),
            f"abs mean DCT ({color_space.upper()}))",
            show=False,
            save=(
                outdir
                / f"{config['name']}{config['variant']}_dct_{color_space}_abs_{mode}.png"
            ),
            log=True,
        )


def get_accuracies(outdir: Path):
    """Run probing and gather results from a series of LVC configurations."""

    # This function is outdated
    assert False

    # Avoid mysterious  reproducibility error by setting CUBLAS_WORKSPACE_CONFIG=:4096:8
    # or CUBLAS_WORKSPACE_CONFIG=:16:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Models to test
    model_configs = [
        {
            "name": "fastseg",
            "variant": "small",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "fastseg",
            "variant": "large",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/large/best_checkpoint_ep172.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "yolov8",
            "variant": "n",
            "snapshot": "yolov8n.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "s",
            "snapshot": "yolov8s.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "l",
            "snapshot": "yolov8l.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
    ]

    grad_type = "dist_sq"

    # sionna_config: SionnaConfig = { "nbits_per_sym": 2, "coderate": 1.0 }
    # sionna_config: SionnaConfig = { "nbits_per_sym": 2, "coderate": 0.5 }
    sionna_config = None

    # Which parts of LVC to optimize with gradients
    grad_tasks = [
        # {
        #     "type": grad_type,
        #     "w": True,
        #     "grad_norm_select": False,
        #     "grad_norm_allocate": False,
        # },
        # {
        #     "type": grad_type,
        #     "w": False,
        #     "grad_norm_select": True,
        #     "grad_norm_allocate": False,
        # },
        # {
        #     "type": grad_type,
        #     "w": False,
        #     "grad_norm_select": False,
        #     "grad_norm_allocate": True,
        # },
        {
            "type": grad_type,
            "g_yuv": True,
            "g_select": True,
            "g_allocate": True,
        },
    ]

    # LVC parameters to test
    color_space = "yuv"
    yuv_mode = [444]
    # The CRs are selected such that at YUV420/64 chunks, the first value selects 3
    # chunks. The rest are multiples of 2. YUV444/64 -> YUV420/256 -> YUV444/256
    # progressively double the amount of chunks, therefore, e.g., the number of sent
    # chunks at CR 0.5 at YUV420/64 corresponds to CR 0.25 at YUV444/64. This makes
    # the CRs and number of chunks directly comparable between different settings.
    CR = [0.03125, 0.06250, 0.12500, 0.25000, 0.50000, 1.00000]
    # CR = np.logspace(-5, 0, 21, base=2)  # 3 samples between two fractions of two
    csnr_dbs = [15]  # [0, 5, 10, 20, 30, "inf"]
    nchunks = [64, 256, 1024]  # [64, 256, 1024]
    estimators = ["zf", "llse"]
    lvc_params = []
    for estimator, mode, nc, csnr_db, cr in itertools.product(
        estimators, yuv_mode, nchunks, csnr_dbs, CR
    ):
        # full-frame DCT
        lvc_params.append(
            {
                "packet_loss": None,
                "seed": 42,
                "mode": mode,
                "cr": cr,
                "csnr_db": csnr_db,
                "estimator": estimator,
                "nchunks": nc,
                "color_space": color_space,
            }
        )

        # block-based DCT
        lvc_params.append(
            {
                "packet_loss": None,
                "seed": 42,
                "mode": mode,
                "cr": cr,
                "csnr_db": csnr_db,
                "estimator": estimator,
                "nchunks": nc,
                "color_space": color_space,
                "dct_w": int(math.sqrt(nc)),
                "dct_h": int(math.sqrt(nc)),
                "grouping": "horizontal_uv",
            }
        )

    all_configs = list(itertools.product(lvc_params, grad_tasks, model_configs))

    print(
        f"=== TOTAL NUMBER OF PERMUTATIONS: {len(model_configs)} + {len(all_configs)} ==="
    )
    seen = set()
    duplicates = []
    for lvcp, task, conf in all_configs:
        key = (tuple(conf.items()), tuple(lvcp.items()), tuple(task.items()))
        if key in seen:
            duplicates.append((conf, lvcp, task))
        else:
            seen.add(key)

    if len(duplicates) != 0:
        print(f"ERROR: Found {len(duplicates)} duplicate entries!")
        return

    num_batches = None
    batch_size = 8
    num_batches_probe = 32
    batch_size_probe = 1

    do_print = False
    (outdir / "partial").mkdir()
    dry = False
    reprobe = False
    reprobe_g = False

    # Setup for parallel processing (ngpus parallel processes)
    total_ngpus = 8  # total number of GPUs on the server
    ngpus = 4  # how many GPUs to use
    gpu_off = 0  # which GPU to start from
    ncpus = int(cpu_count() * ngpus / total_ngpus)
    ngpus = min(len(all_configs), ngpus)

    model_config_groups = np.array_split(model_configs, ngpus)
    config_groups = np.array_split(all_configs, ngpus)

    grad_type_groups = [grad_type] * ngpus
    possible_nchunks_groups = [nchunks] * ngpus
    cpus = np.array_split(range(ncpus), ngpus)
    devices = [f"cuda:{i+gpu_off}" for i in range(ngpus)]
    color_spaces = [color_space] * ngpus
    ranks = range(gpu_off, gpu_off + ngpus)
    num_workers_groups = [0] * ngpus
    num_batches_groups = [num_batches] * ngpus
    batch_size_groups = [batch_size] * ngpus
    num_batches_probe_groups = [num_batches_probe] * ngpus
    batch_size_probe_groups = [batch_size_probe] * ngpus
    do_print_groups = [do_print] * ngpus
    outdirs = [outdir / "partial"] * ngpus
    dries = [dry] * ngpus
    reprobes = [reprobe] * ngpus
    reprobes_g = [reprobe_g] * ngpus
    sionna_configs = [sionna_config] * ngpus

    with Pool(processes=ngpus) as pool:
        probe_result_groups = pool.starmap(
            probe_models,
            zip(
                model_config_groups,
                grad_type_groups,
                possible_nchunks_groups,
                devices,
                color_spaces,
                ranks,
                cpus,
                num_workers_groups,
                num_batches_groups,
                batch_size_groups,
                num_batches_probe_groups,
                batch_size_probe_groups,
                do_print_groups,
                outdirs,
                dries,
            ),
        )

    probe_results = {}
    for probe_result_group in probe_result_groups:
        probe_results.update(probe_result_group)

    print("Got probe results from models", ", ".join(probe_results.keys()))
    if not dry:
        fname = outdir / "probe_results.pt"
        torch.save(probe_results, fname)

    probe_result_groups_all = [probe_results] * ngpus

    with Pool(processes=ngpus) as pool:
        lvc_result_groups = pool.starmap(
            run_models,
            zip(
                config_groups,
                probe_result_groups_all,
                devices,
                color_spaces,
                ranks,
                cpus,
                num_workers_groups,
                num_batches_groups,
                batch_size_groups,
                num_batches_probe_groups,
                batch_size_probe_groups,
                do_print_groups,
                outdirs,
                dries,
                reprobes,
                reprobes_g,
                sionna_configs,
            ),
        )

    # # results_groups.sort(key=lambda x: (x["lvc_params"]["cr"], x["lvc_params"]["csnr_db"]))
    lvc_results = []
    for lvc_result_group in lvc_result_groups:
        lvc_results.extend(lvc_result_group)

    print("Done")
    if not dry:
        fname = outdir / "lvc_results.pt"
        torch.save(lvc_results, fname)


def get_accuracies_single(outdir: Path):
    """Run probing and gather results from a series of LVC configurations.

    Uses only a single process / GPU
    """

    # Avoid mysterious  reproducibility error by setting CUBLAS_WORKSPACE_CONFIG=:4096:8
    # or CUBLAS_WORKSPACE_CONFIG=:16:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # For going around non-deterministic NLLLoss error in fastseg
    torch.use_deterministic_algorithms(False)

    # Limit processing to one GPU
    gpu_i = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_i}"

    # Models to test
    model_configs_single: List[ModelConfig] = [
        # {
        #     "name": "fastseg",
        #     "variant": "small",
        #     "snapshot": (
        #         Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
        #     ),
        #     "unit": "mean_iu",
        # },
        # {
        #     "name": "fastseg",
        #     "variant": "large",
        #     "snapshot": (
        #         Path.home() / "data/models/fastseg/raw/large/best_checkpoint_ep172.pth"
        #     ),
        #     "unit": "mean_iu",
        # },
        # {
        #     "name": "yolov8",
        #     "variant": "n",
        #     "snapshot": "yolov8n.pt",
        #     "unit": "mAP_50_95",
        #     "task": "detect",
        #     "data_file": "coco.yaml",
        # },
        # {
        #     "name": "yolov8",
        #     "variant": "s",
        #     "snapshot": "yolov8s.pt",
        #     "unit": "mAP_50_95",
        #     "task": "detect",
        #     "data_file": "coco.yaml",
        # },
        {
            "name": "yolov8",
            "variant": "l",
            "snapshot": "yolov8l.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
    ]

    model_configs_double: List[Tuple[ModelConfig, ModelConfig]] = list(
        itertools.product(model_configs_single, model_configs_single)
    )

    model_configs_probe = model_configs_single
    model_configs_eval = model_configs_single  # or _double

    grad_type = "dist_sq"
    g_opt = True  # Enable GRACE optimizations

    # Probe config:
    dct_size = None  # (8, 8)
    subtract_mean = True
    scale = 1.0
    norm = "sq"

    sionna_configs: List[SionnaConfig | None] = [
        # Analog transmission (CV-Cast):
        {"nbits_per_sym": 2, "coderate": 1.0}
        # Digital transmission (JPEG, GRACE):
        # {"nbits_per_sym": 2, "coderate": 0.5},
        # {"nbits_per_sym": 4, "coderate": 0.5},
        # {"nbits_per_sym": 6, "coderate": 0.5},
        # No Sionna channel (=> use basic AWGN channel)
        # None,
    ]

    # Which parts of LVC to optimize with gradients
    grad_tasks = [
        # {
        #     "type": grad_type,
        #     "w": True,
        #     "grad_norm_select": False,
        #     "grad_norm_allocate": False,
        # },
        # {
        #     "type": grad_type,
        #     "w": False,
        #     "grad_norm_select": True,
        #     "grad_norm_allocate": False,
        # },
        # {
        #     "type": grad_type,
        #     "w": False,
        #     "grad_norm_select": False,
        #     "grad_norm_allocate": True,
        # },
        {
            "type": grad_type,
            "g_yuv": True,
            "g_select": True,
            "g_allocate": True,
        },
    ]

    # LVC parameters to test
    color_space = "yuv"
    yuv_mode = [444]
    # The CRs are selected such that at YUV420/64 chunks, the first value selects 3
    # chunks. The rest are multiples of 2. YUV444/64 -> YUV420/256 -> YUV444/256
    # progressively double the amount of chunks, therefore, e.g., the number of sent
    # chunks at CR 0.5 at YUV420/64 corresponds to CR 0.25 at YUV444/64. This makes
    # the CRs and number of chunks directly comparable between different settings.
    # CR = [0.03125, 0.06250, 0.12500, 0.25000, 0.50000, 1.00000]
    CR = [1.0]
    # CR = np.logspace(-5, 0, 21, base=2)  # 3 samples between two fractions of two
    # csnr_dbs = ["inf"]
    # csnr_dbs = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    csnr_dbs = [2.5, 7.5]
    # csnr_dbs = [0, 1.25, 3.75, 6.25, 8.75]
    nchunks = [256]  # [64, 256, 1024]
    estimators = ["zf"]  # , "llse"]
    lvc_params = []
    for estimator, mode, nc, csnr_db, cr in itertools.product(
        estimators, yuv_mode, nchunks, csnr_dbs, CR
    ):
        # full-frame DCT
        lvc_params.append(
            {
                "packet_loss": None,
                "seed": 42,
                "mode": mode,
                "cr": cr,
                "csnr_db": csnr_db,
                "estimator": estimator,
                "nchunks": nc,
                "color_space": color_space,
            }
        )

        # block-based DCT
        # lvc_params.append(
        #     {
        #         "packet_loss": None,
        #         "seed": 42,
        #         "mode": mode,
        #         "cr": cr,
        #         "csnr_db": csnr_db,
        #         "estimator": estimator,
        #         "nchunks": nc,
        #         "color_space": color_space,
        #         "dct_w": int(math.sqrt(nc)),
        #         "dct_h": int(math.sqrt(nc)),
        #         "grouping": "horizontal_uv",
        #     }
        # )

    Q_SEARCH_FILES: List[str | Path] = [
        # normsq:
        # "experiments_tupu/runs/run21_keep/q_search_fastseg_small_Noneimgs.pt",
        # "experiments_tupu/runs/run22_keep/q_search_fastseg_large_Noneimgs.pt",
        # "experiments_hupu/runs/run49_keep/q_search_yolov8_n_Noneimgs.pt",
        # "experiments_hupu/runs/run50_keep/q_search_yolov8_s_Noneimgs.pt",
        # "experiments_hupu/runs/run51_keep/q_search_yolov8_l_Noneimgs.pt",
        # normabs:
        # "experiments_hupu/runs/run57_keep/q_search_fastseg_small_Noneimgs.pt",
        # "experiments_hupu/runs/run58_keep/q_search_fastseg_large_Noneimgs.pt",
        # "experiments_hupu/runs/run59_keep/q_search_yolov8_n_Noneimgs.pt",
        # "experiments_hupu/runs/run60_keep/q_search_yolov8_s_Noneimgs.pt",
        # "experiments_hupu/runs/run61_keep/q_search_yolov8_l_Noneimgs.pt",
        # all configs, 8 images:
        "experiments_tupu/runs/run27_keep/q_search_fastseg_small_8imgs.pt",
        "experiments_tupu/runs/run28_keep/q_search_fastseg_large_8imgs.pt",
        "experiments_tupu/runs/run29_keep/q_search_yolov8_n_8imgs.pt",
        "experiments_tupu/runs/run30_keep/q_search_yolov8_s_8imgs.pt",
        "experiments_tupu/runs/run31_keep/q_search_yolov8_l_8imgs.pt",
    ]

    q_search_filters: QSearchFilters = {
        "mode": mode,
        "dist": grad_type,
        "norm": norm,
        "sub": "submean" if subtract_mean else "nosubmean",
        "dctsz": "ff" if dct_size is None else f"bb{dct_size[0]}x{dct_size[1]}",
        "sc": scale,
    }

    turbojpeg_enc = False
    turbojpeg_dec = False
    codec = "tcm"

    forced_params = [
        # MODEL_FILES["tcm_n64_lambda0.0025"],
        # MODEL_FILES["tcm_n64_lambda0.0035"],
        # MODEL_FILES["tcm_n64_lambda0.0067"],
        # MODEL_FILES["tcm_n64_lambda0.013"],
        # MODEL_FILES["tcm_n64_lambda0.025"],
        # MODEL_FILES["tcm_n64_lambda0.05"],
        # MODEL_FILES["tcm_n128_lambda0.05"],
    ]

    # jpeg_init_params = [
    #     {
    #         "codec": codec,
    #         "turbojpeg_enc": False,
    #         "turbojpeg_dec": False,
    #         "forced_param": forced_param,
    #     }
    #     for forced_param in forced_params
    # ]

    jpeg_init_params = [
        None,
        #     # {
        #     #     "codec": "jpeg",
        #     #     "turbojpeg_enc": turbojpeg_enc,
        #     #     "turbojpeg_dec": turbojpeg_dec,
        #     #     "forced_param": None,
        #     # },
        #     # {
        #     #     "codec": "grace",
        #     #     "turbojpeg_enc": turbojpeg_enc,
        #     #     "turbojpeg_dec": turbojpeg_dec,
        #     #     "forced_param": None,
        #     # },
    ]

    all_configs = list(
        itertools.product(
            lvc_params, grad_tasks, model_configs_eval, jpeg_init_params, sionna_configs
        )
    )

    print(
        f"=== TOTAL NUMBER OF PERMUTATIONS: probe {len(model_configs_single)} + eval {len(all_configs)} ==="
    )
    seen = set()
    duplicates = []
    for lvcp, task, model_conf, jpeg_init_param, sionna_conf in all_configs:
        if isinstance(model_conf, tuple):
            model_conf_key = (
                tuple(model_conf[0].items()),
                tuple(model_conf[1].items()),
            )
        else:
            model_conf_key = tuple(model_conf.items())

        key = (
            model_conf_key,
            tuple(lvcp.items()),
            tuple(task.items()),
            (
                jpeg_init_param
                if jpeg_init_param is None
                else frozenset(jpeg_init_param.items())
            ),
            sionna_conf if sionna_conf is None else frozenset(sionna_conf.items()),
        )

        if key in seen:
            duplicates.append((model_conf, lvcp, task, jpeg_init_param, sionna_conf))
        else:
            seen.add(key)

    if len(duplicates) != 0:
        print(f"ERROR: Found {len(duplicates)} duplicate entries!")
        pprint.pp(duplicates)
        return

    num_batches = None
    batch_size = 1
    num_batches_probe = 32
    batch_size_probe = 1

    do_print = False
    (outdir / "partial").mkdir()
    dry = False
    reprobe = False
    reprobe_g = False

    device = "cuda:0"
    rank = 0
    total_ngpus = 8
    ncpus = int(cpu_count() / total_ngpus)
    cpus = np.array(range(gpu_i * ncpus, gpu_i * ncpus + ncpus))

    probe_results = probe_models(
        model_configs_probe,
        grad_type,
        nchunks,
        device,
        color_space,
        rank,
        cpus,
        num_workers=0,
        num_batches=num_batches,
        batch_size=batch_size,
        num_batches_probe=num_batches_probe,
        batch_size_probe=batch_size_probe,
        dct_size=dct_size,
        subtract_mean=subtract_mean,
        scale=scale,
        norm=norm,
        do_print=do_print,
        outdir=outdir / "partial",
        dry=dry,
    )

    # # load probe from file
    # model_id = get_model_id(model_configs_single[0])
    # probe_fname = f"probe_result_full_{model_id}_{mode}_{grad_type}_{'submean' if subtract_mean else 'nosubmean'}_{'ff' if dct_size is None else 'bb{}x{}'.format(dct_size[0], dct_size[1])}_sc{scale:.0f}_norm{norm}.pt"
    # print(probe_fname)
    # probe_data = torch.load(f"experiments/runs/run24_keep/{probe_fname}")
    # probe_data["w_yuv"] = probe_data["W"]
    # probe_results = {model_id: probe_data}

    print("Got probe results from models", ", ".join(probe_results.keys()))
    if not dry:
        fname = outdir / "probe_results.pt"
        torch.save(probe_results, fname)

    lvc_results = run_models(
        all_configs,
        probe_results,
        device,
        color_space,
        rank,
        cpus,
        num_workers=0,
        num_batches=num_batches,
        batch_size=batch_size,
        num_batches_probe=num_batches_probe,
        batch_size_probe=batch_size_probe,
        do_print=do_print,
        outdir=outdir / "partial",
        dry=dry,
        reprobe=reprobe,
        reprobe_g=reprobe_g,
        g_opt=g_opt,
        grad_type=grad_type,
        q_search_files=Q_SEARCH_FILES,
        q_search_filters=q_search_filters,
    )

    print("Done")
    if not dry:
        fname = outdir / "lvc_results.pt"
        torch.save(lvc_results, fname)


def eval_metrics(outdir: Path):
    print("Getting PSNR")

    # Avoid mysterious  reproducibility error by setting CUBLAS_WORKSPACE_CONFIG=:4096:8
    # or CUBLAS_WORKSPACE_CONFIG=:16:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Setup
    torch.manual_seed(SEED)
    gpu_i = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_i}"
    device = set_device(f"cuda:0")
    cpus = set(range(84, 96))
    os.sched_setaffinity(0, cpus)
    affinity = os.sched_getaffinity(0)
    print(f"Running on CPUs: {affinity}")
    torch.set_num_threads(len(cpus))

    # Models to test
    model_configs = [
        {
            "name": "fastseg",
            "variant": "small",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "fastseg",
            "variant": "large",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/large/best_checkpoint_ep172.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "yolov8",
            "variant": "n",
            "snapshot": "yolov8n.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "s",
            "snapshot": "yolov8s.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "l",
            "snapshot": "yolov8l.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
    ]

    # LVC parameters to test
    color_space = "yuv"
    yuv_mode = [444]
    # The CRs are selected such that at YUV420/64 chunks, the first value selects 3
    # chunks. The rest are multiples of 2. YUV444/64 -> YUV420/256 -> YUV444/256
    # progressively double the amount of chunks, therefore, e.g., the number of sent
    # chunks at CR 0.5 at YUV420/64 corresponds to CR 0.25 at YUV444/64. This makes
    # the CRs and number of chunks directly comparable between different settings.
    CR = [0.03125, 0.06250, 0.12500, 0.25000, 0.50000, 1.00000]
    # CR = np.logspace(-5, 0, 21, base=2)  # 3 samples between two fractions of two
    csnr_dbs = [0, 5, 10, 20, 30, "inf"]
    nchunks = [64, 256, 1024]
    estimators = ["zf", "llse"]
    lvc_params = []
    for estimator, mode, nc, csnr_db, cr in itertools.product(
        estimators, yuv_mode, nchunks, csnr_dbs, CR
    ):
        # full-frame DCT
        lvc_params.append(
            {
                "packet_loss": None,
                "seed": 42,
                "mode": mode,
                "cr": cr,
                "csnr_db": csnr_db,
                "estimator": estimator,
                "nchunks": nc,
                "color_space": color_space,
            }
        )

        # block-based DCT
        lvc_params.append(
            {
                "packet_loss": None,
                "seed": 42,
                "mode": mode,
                "cr": cr,
                "csnr_db": csnr_db,
                "estimator": estimator,
                "nchunks": nc,
                "color_space": color_space,
                "dct_w": int(math.sqrt(nc)),
                "dct_h": int(math.sqrt(nc)),
                "grouping": "horizontal_uv",
            }
        )

    all_configs = list(itertools.product(lvc_params, model_configs))

    num_batches = None
    batch_size = 8
    num_batches_probe = 32
    batch_size_probe = 1
    dry = False

    (outdir / "partial").mkdir()

    # First, probe models for gradients
    probe_results = torch.load("experiments/run24_keep/probe_results.pt")
    # probe_results = probe_models(
    #     model_configs,
    #     nchunks,
    #     device,
    #     color_space,
    #     0,
    #     None,
    #     num_batches=num_batches,
    #     batch_size=batch_size,
    #     num_batches_probe=num_batches_probe,
    #     batch_size_probe=batch_size_probe,
    #     do_print=False,
    #     outdir=outdir / "partial",
    #     dry=False,
    # )

    results = []

    grad_task = {
        "type": "dist_sq",
        "g_yuv": True,
        "g_select": True,
        "g_allocate": True,
    }

    # Then run the metrics eval
    for i, (lvc_param_dict, model_config) in enumerate(all_configs):
        start_time = time.perf_counter()

        model_id = get_model_id(model_config)
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
        msg = (
            f"({i:3d}/{len(all_configs):3d}), {model_id:15s},"
            + f" {est:4s}, {mode:3d}, {nchunks:4d}, {csnr_db:3s} dB, CR {cr:.4f}, {blk:2s}, {task_msg:11s}"
        )
        print(f"---- {msg}        (start) ----")

        grad_yuv_key = "grads_yuv_420" if mode == 420 else "grads_yuv"
        grad_key = "grads_norm_420" if mode == 420 else "grads_norm"
        probe_result = probe_results[model_id]
        grads_mean = probe_result[grad_yuv_key].to(device, non_blocking=True)
        grads_norm = probe_result[grad_key][nchunks].to(device, non_blocking=True)
        w_yuv = probe_result["w_yuv"].to(device, non_blocking=True)

        # Create models with default and grad-optimized LVC
        grad_config: GradConfig = {
            "type": grad_task["type"],
            "g_yuv": grad_task["g_yuv"],
            "g_select": grad_task["g_select"],
            "g_allocate": grad_task["g_allocate"],
            "w": w_yuv,
            "grad_mean": grads_mean,
            "grad_norm": grads_norm,
        }

        lvc_chain = create_lvc(
            lvc_param_dict,
            device,
            False,
            unsqueeze=True,
            do_print=False,
            grad_config=None,
        )

        lvc_chain_g = create_lvc(
            lvc_param_dict,
            device,
            False,
            unsqueeze=True,
            do_print=False,
            grad_config=grad_config,
        )

        model_lvc = get_model(
            model_config,
            device,
            num_batches,
            batch_size,
            lvc_chain=lvc_chain,
            color_space=color_space,
            do_print=False,
        )

        model_lvc_g = get_model(
            model_config,
            device,
            num_batches,
            batch_size,
            lvc_chain=lvc_chain_g,
            color_space=color_space,
            do_print=False,
        )

        # Get reference model
        model_ref = get_model(
            model_config,
            device,
            num_batches,
            batch_size,
            lvc_chain=None,
            color_space=color_space,
            do_print=False,
        )

        # Calculate metrics
        metrics_lvc = model_lvc.eval_metrics(model_ref, do_print=False)
        metrics_lvc_g = model_lvc_g.eval_metrics(model_ref, do_print=False)

        partial_res = {
            "model_id": model_id,
            "lvc_params": lvc_param_dict,
            "color_space": color_space,
            "grad_tasks": grad_task,
            "lvc": metrics_lvc,
            "lvc_g": metrics_lvc_g,
        }
        results.append(partial_res)

        fname = outdir / "partial" / f"metrics_{i:04d}.pt"
        if not dry:
            torch.save(partial_res, fname)
            with open(fname.with_suffix(".json"), "w") as wf:
                json.dump(partial_res, wf)

        duration = int(time.perf_counter() - start_time)
        duration_msg = f"{duration // 60:02d}m{duration % 60:02d}s"
        psnr_msg = f"PSNR {metrics_lvc['psnr']['mean']:5.2f}dB (G {metrics_lvc_g['psnr']['mean']:5.2f}dB)"
        print(f"==== {msg} {duration_msg} {psnr_msg} (end) -> {fname} ====")

    if not dry:
        fname = outdir / "metrics.pt"
        torch.save(results, fname)
        with open(fname.with_suffix(".json"), "w") as wf:
            json.dump(results, wf)


def predict(outdir: Path):
    print("Predict")
    # Avoid mysterious  reproducibility error by setting CUBLAS_WORKSPACE_CONFIG=:4096:8
    # or CUBLAS_WORKSPACE_CONFIG=:16:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Setup
    torch.manual_seed(SEED)
    gpu_i = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_i}"
    total_ngpus = 8
    device = set_device(f"cuda:0")
    ncpus = int(cpu_count() / total_ngpus)
    cpus = range(gpu_i * ncpus, gpu_i * ncpus + ncpus)
    os.sched_setaffinity(0, cpus)
    affinity = os.sched_getaffinity(0)
    print(f"Running on CPUs: {affinity}")
    torch.set_num_threads(len(cpus))

    model_config: ModelConfig = {
        "name": "fastseg",
        "variant": "small",
        "snapshot": (
            Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
        ),
        "unit": "mean_iu",
    }

    Q_SEARCH_FILES = {
        "fastseg_small": "experiments_tupu/runs/run21_keep/q_search_fastseg_small_Noneimgs.pt",
        "fastseg_large": "experiments_tupu/runs/run22_keep/q_search_fastseg_large_Noneimgs.pt",
        "yolov8_n": "experiments_hupu/runs/run49_keep/q_search_yolov8_n_Noneimgs.pt",
        "yolov8_s": "experiments_hupu/runs/run50_keep/q_search_yolov8_s_Noneimgs.pt",
        "yolov8_l": "experiments_hupu/runs/run51_keep/q_search_yolov8_l_Noneimgs.pt",
    }

    num_batches = 4
    batch_size = 1

    color_space = "yuv"
    mode = 444
    cr = 1.0
    csnr_db = 2.5  # "inf"
    estimator = "zf"
    nc = nchunks = 256

    lvc_param_dict = {
        "packet_loss": None,
        "seed": 42,
        "mode": mode,
        "cr": cr,
        "csnr_db": csnr_db,
        "estimator": estimator,
        "nchunks": nc,
        "color_space": color_space,
        # "dct_w": int(math.sqrt(nc)),
        # "dct_h": int(math.sqrt(nc)),
        # "grouping": "vertical_uv",
    }

    probe_results = torch.load("probe_results.pt")

    model_id = get_model_id(model_config)
    grad_key = "grads_norm_420" if mode == 420 else "grads_norm"
    grad_yuv_key = "grads_yuv_420" if mode == 420 else "grads_yuv"
    probe_result = probe_results[model_id]
    grads_norm = probe_result[grad_key][nchunks].to(device, non_blocking=True)
    grads_mean = probe_result[grad_yuv_key].to(device, non_blocking=True)
    w_yuv = probe_result["w_yuv"].to(device, non_blocking=True)

    # sionna_config: SionnaConfig = {"nbits_per_sym": 2, "coderate": 1.0}
    sionna_config: SionnaConfig = {"nbits_per_sym": 2, "coderate": 0.5}
    # sionna_config = None

    codec = "tcm"  # None

    if codec in ["jpeg", "grace"]:
        param = fetch_param(
            pd.DataFrame(torch.load(Q_SEARCH_FILES[model_id])),
            model_id,
            codec,
            sionna_config["nbits_per_sym"],
            cr,
            DEFAULT_FILTERS,
        )
    elif codec == "icm":
        param = MODEL_FILES["icm"]
    elif codec == "tcm":
        param = MODEL_FILES["tcm_n64_lambda0.0067"]
    else:
        param = None

    print(f"Codec: {codec}, param: {param}")

    if codec == "grace":
        jpeg_config: JPEGConfig = {
            "codec": "grace",
            "param": param,
            "param_name": "B",
            "param_fmt": "{:12.5e}",
            "turbojpeg_enc": False,
            "turbojpeg_dec": False,
        }
    elif codec == "jpeg":
        jpeg_config: JPEGConfig = {
            "codec": "jpeg",
            "param": param,
            "param_name": "Q",
            "param_fmt": "{:3}",
            "turbojpeg_enc": False,
            "turbojpeg_dec": False,
        }
    elif codec in ["icm", "tcm"]:
        jpeg_config: JPEGConfig = {
            "codec": codec,
            "param": param,
            "param_name": "model",
            "param_fmt": " {}",
            "turbojpeg_enc": False,
            "turbojpeg_dec": False,
        }
    else:
        jpeg_config = None

    # custom_dataset_dir = "experiments/test_data"
    custom_dataset_dir = None

    grad_config: GradConfig = {
        "type": "dist_sq",
        "g_yuv": True,
        "g_select": True,
        "g_allocate": True,
        "w": w_yuv,
        "grad_mean": grads_mean,
        "grad_norm": grads_norm,
    }

    do_print = True

    # Default LVC (GRACE can't run without grad-optimization)
    if jpeg_config is None or jpeg_config["codec"] != "grace":
        print("=== Predicting LVC ===")

        lvc_chain = create_lvc(
            lvc_param_dict,
            device,
            False,
            unsqueeze=True,
            do_print=do_print,
            grad_config=None,
            sionna_config=sionna_config,
            jpeg_config=jpeg_config,
        )

        model_lvc = get_model(
            model_config,
            device,
            num_batches,
            batch_size,
            lvc_chain=lvc_chain,
            color_space=color_space,
            do_print=do_print,
        )

        (
            colorized_lvc,
            blended_lvc,
            origs_lvc,
            gt_lvc,
        ) = model_lvc.predict(custom_dataset_dir)

        for i, (col, blend, orig, gt) in enumerate(
            zip(colorized_lvc, blended_lvc, origs_lvc, gt_lvc)
        ):
            orig.save(outdir / f"orig_lvc_{i:03d}.png")
            col.save(outdir / f"colorized_lvc_{i:03d}.png")
            blend.save(outdir / f"blend_lvc_{i:03d}.png")
            gt.save(outdir / f"gt_lvc_{i:03d}.png")

    # CV-Cast (JPEG can't be grad-optimized)
    if jpeg_config is None or jpeg_config["codec"] not in ["jpeg", "icm"]:
        print("=== Predicting CV-Cast ===")

        lvc_chain_g = create_lvc(
            lvc_param_dict,
            device,
            False,
            unsqueeze=True,
            do_print=do_print,
            grad_config=grad_config,
            sionna_config=sionna_config,
            jpeg_config=jpeg_config,
        )

        model_lvc_g = get_model(
            model_config,
            device,
            num_batches,
            batch_size,
            lvc_chain=lvc_chain_g,
            color_space=color_space,
            do_print=do_print,
        )

        (
            colorized_lvc_g,
            blended_lvc_g,
            origs_lvc_g,
            gt_lvc_g,
        ) = model_lvc_g.predict(custom_dataset_dir)

        for i, (col, blend, orig, gt) in enumerate(
            zip(colorized_lvc_g, blended_lvc_g, origs_lvc_g, gt_lvc_g)
        ):
            orig.save(outdir / f"orig_lvc_g_{i:03d}.png")
            col.save(outdir / f"colorized_lvc_g_{i:03d}.png")
            blend.save(outdir / f"blend_lvc_g_{i:03d}.png")
            gt.save(outdir / f"gt_lvc_g_{i:03d}.png")

    # Reference (no coding or transmission)
    print("=== Predicting Reference ===")

    model_ref = get_model(
        model_config,
        device,
        num_batches,
        batch_size,
        lvc_chain=None,
        color_space=color_space,
        do_print=do_print,
    )

    (
        colorized_ref,
        blended_ref,
        origs_ref,
        gt_ref,
    ) = model_ref.predict(custom_dataset_dir)

    for i, (col, blend, orig, gt) in enumerate(
        zip(colorized_ref, blended_ref, origs_ref, gt_ref)
    ):
        orig.save(outdir / f"orig_ref_{i:03d}.png")
        col.save(outdir / f"colorized_ref_{i:03d}.png")
        blend.save(outdir / f"blend_ref_{i:03d}.png")
        gt.save(outdir / f"gt_ref_{i:03d}.png")


def probe_only(outdir: Path):
    """Run only probe, save the results to .pt files"""

    print("Probe")
    torch.manual_seed(SEED)
    gpu_i = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_i}"
    device = set_device(f"cuda:0")
    cpus = set(range(84, 96))
    os.sched_setaffinity(0, cpus)
    affinity = os.sched_getaffinity(0)
    print(f"Running on CPUs: {affinity}")
    torch.set_num_threads(len(cpus))

    color_space = "yuv"
    modes = [444, 420]
    dct_sizes = [None, (8, 8), (16, 16)]
    possible_nchunks = [64, 256, 1024]
    subtract_means = [True, False]
    grad_types = ["dist_sq", "dist_abs"]
    probe_scales = [1.0, 255.0]
    norms = ["sq", "abs"]
    num_batches_probe = 32
    batch_size_probe = 1

    configs = [
        {
            "name": "fastseg",
            "variant": "small",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "fastseg",
            "variant": "large",
            "snapshot": (
                Path.home() / "data/models/fastseg/raw/large/best_checkpoint_ep172.pth"
            ),
            "unit": "mean_iu",
        },
        {
            "name": "yolov8",
            "variant": "n",
            "snapshot": "yolov8n.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "s",
            "snapshot": "yolov8s.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "yolov8",
            "variant": "l",
            "snapshot": "yolov8l.pt",
            "unit": "mAP_50_95",
            "task": "detect",
            "data_file": "coco.yaml",
        },
        {
            "name": "drn",
            "variant": "d_22",
            "snapshot": Path.home() / "data/models/drn/drn_d_22_cityscapes.pth",
            "unit": "mean_iu",
        },
        # {
        #     "name": "drn",
        #     "variant": "d_38",
        #     "snapshot": Path.home() / "data/models/drn/drn_d_38_cityscapes.pth",
        #     "unit": "mean_iu",
        # },
    ]

    for (
        config,
        mode,
        grad_type,
        subtract_mean,
        dct_size,
        scale,
        norm,
    ) in itertools.product(
        configs, modes, grad_types, subtract_means, dct_sizes, probe_scales, norms
    ):
        model_probe = get_model(
            config, device, num_batches_probe, batch_size_probe, color_space=color_space
        )

        result_probe = model_probe.run_probe(
            grad_type,
            possible_nchunks,
            dct_size=dct_size,
            subtract_mean=subtract_mean,
            scale=scale,
            norm=norm,
        )

        model_id = get_model_id(config)
        submsg = "submean" if subtract_mean else "nosubmean"
        dctmsg = "ff" if dct_size is None else f"bb{dct_size[1]}x{dct_size[0]}"
        outfile = (
            outdir
            / f"probe_result_full_{model_id}_{mode}_{grad_type}_{submsg}_{dctmsg}_sc{scale:.0f}_norm{norm}.pt"
        )
        print(f"saving output to {outfile}")
        torch.save(result_probe, outfile)


def q_search(outdir: Path):
    """Searching for a JPEG/GRACE quality parameter to meet LCT-equivalent compression
    ratio considering m-QAM modulation."""

    print("q_search")

    MODELS: List[
        Literal["fastseg_small", "fastseg_large", "yolov8_n", "yolov8_s", "yolov8_l"]
    ] = ["fastseg_small", "fastseg_large", "yolov8_n", "yolov8_s", "yolov8_l"]

    nmodel = 4
    model = MODELS[nmodel]

    CPUS_PER_MODEL = 6
    START_CPU = 18 + nmodel * CPUS_PER_MODEL

    # Run on one core only; Does not seem to affect, run with:
    # numactl --physcpubind=+1 python run.py
    cpus = range(START_CPU, START_CPU + CPUS_PER_MODEL)
    os.sched_setaffinity(0, cpus)
    affinity = os.sched_getaffinity(0)
    print(f"Running on CPUs: {affinity}")
    torch.set_num_threads(len(cpus))

    # Directory with probe results generated by probe_only()
    PROBE_DIR = Path(
        # "/home/jakub/git/nn-spectral-sensitivity/experiments_tupu/runs/run10_keep"
        "/home/jakub/git/nn-spectral-sensitivity/experiments_tupu/runs/run24_keep"
    )

    DCT_SIZE = Size(h=8, w=8)

    # Static search params
    NBITS_PER_SYM = [2, 4, 6]  # 4-, 16- and 64-QAM
    TARGET_CRS = [0.03125, 0.06250, 0.12500, 0.25000, 0.50000, 1.00000]
    NIMAGES = None

    # Probe params
    codecs = ["jpeg"]  # jpeg or grace
    modes = [444]
    dists = ["dist_sq"]  # dist_abs or dist_sq
    norms = ["sq"]  # abs or sq
    subs = ["submean"]  # submean or nosubmean
    dctszs = ["ff"]  # bb8x8 or ff
    scs = [1]  # 1 or 255

    final_res = []
    do_save = True
    do_print = False

    if do_save:
        partial_outdir = outdir
    else:
        partial_outdir = None

    args = list(
        itertools.product(
            modes, dists, norms, subs, dctszs, scs, codecs, NBITS_PER_SYM, TARGET_CRS
        )
    )
    ntotal = len(args)

    print(f"Model: {model}, total combinations: {ntotal}")
    for i, (
        mode,
        dist,
        norm,
        sub,
        dctsz,
        sc,
        codec,
        nbits_per_sym,
        target_cr,
    ) in enumerate(args, start=1):
        res = run_qsearch(
            i,
            ntotal,
            codec,
            model,
            mode,
            dist,
            norm,
            sub,
            dctsz,
            sc,
            PROBE_DIR,
            nbits_per_sym,
            target_cr,
            nimages=NIMAGES,
            dct_size=DCT_SIZE,
            outdir=partial_outdir,
            do_print=do_print,
        )
        final_res.append(res)

    if do_save:
        outfile = outdir / f"q_search_{model}_{NIMAGES}imgs.pt"
        print(f"Saving to {outfile}")
        torch.save(final_res, outfile)
        with open(outfile.with_suffix(".json"), "w") as wf:
            print(f"Saving to {outfile.with_suffix('.json')}")
            json.dump(final_res, wf)

    print("Done")


def run_all(tasks: List[str] = ["accuracies"]):
    if not Path(".git").exists():
        raise ValueError("Run from repository root")

    outdir_root = Path("experiments/runs").resolve(strict=True)
    largest_idx = 0
    for d in outdir_root.glob("run*"):
        if d.is_dir():
            try:
                idx = int(d.name[3:])
            except:
                continue

            if idx > largest_idx:
                largest_idx = idx

    outdir = outdir_root / f"run{largest_idx + 1:02d}"
    outdir.mkdir()
    print(f"Output directory: {outdir}")

    # Which experiments to run

    # ... either these two
    if "variance" in tasks:
        gradient_variance(outdir)

    if "maps" in tasks:
        gradient_maps(outdir)

    if "metrics" in tasks:
        eval_metrics(outdir)

    if "predict" in tasks:
        predict(outdir)

    if "accuracies_single" in tasks:
        get_accuracies_single(outdir)

    if "probe_only" in tasks:
        probe_only(outdir)

    if "q_search" in tasks:
        q_search(outdir)

    # ... or this one (not all at once)
    if "accuracies" in tasks:
        get_accuracies(outdir)


if __name__ == "__main__":
    run_all()
