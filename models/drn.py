import itertools
import sys
from pathlib import Path
from typing import Tuple, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import ArrayLike
from PIL import Image

from geffnet import tf_mobilenetv3_large_100, tf_mobilenetv3_small_100
from geffnet.efficientnet_builder import InvertedResidual, Conv2dSame, Conv2dSameExport
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .model import Model, ModelConfig, plot_channels
from .fastseg import probe
from datasets import (
    CITYSCAPES_IGNORE_LABEL,
    CITYSCAPES_NUM_CLASSES,
    CITYSCAPES_MEAN,
    CITYSCAPES_STD,
    get_cityscapes_dataloader,
    get_folder_dataloader,
    cityscapes_labels,
)
from submodules.drn.segment import DRNSeg
from transforms.dct import Dct, Idct
from transforms.color_transforms import RgbToYcbcr, YcbcrToRgb
from utils import set_device
from utils import block_process


class DrnConfig(ModelConfig):
    pass


class DrnModel(Model):
    def __init__(
        self,
        config: DrnConfig,
        device: str | torch.device | None = None,
        num_batches: int | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        lvc_chain: T.Compose | None = None,
        color_space: str = "rgb",
        do_print: bool = False,
    ):
        super().__init__(
            config,
            device=device,
            num_batches=num_batches,
            batch_size=batch_size,
            num_workers=num_workers,
            lvc_chain=lvc_chain,
            color_space=color_space,
            do_print=do_print,
        )

        self._cityscapes_root = Path.home() / "data/cityscapes"
        self._criterion = nn.NLLLoss2d(ignore_index=255).to(device)

        model_id = config["name"] + "_" + config["variant"]
        single_model = DRNSeg(
            model_id,
            classes=CITYSCAPES_NUM_CLASSES,
            pretrained_model=None,
            pretrained=True,
        ).to(device)
        single_model.load_state_dict(torch.load(config["snapshot"]))
        self._model = torch.nn.DataParallel(single_model).to(device)

    def _get_dataloader(self, split: Literal["train", "val"]) -> DataLoader:
        return get_cityscapes_dataloader(
            self._cityscapes_root,
            split,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            lvc_chain=self._lvc_chain,
            shuffle=False,
        )

    def _get_batch_images(self, batch) -> torch.Tensor:
        return batch[0].to(self._device, non_blocking=True)

    def _probe(self, probe_config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        self._dataloader = get_cityscapes_dataloader(
            self._cityscapes_root,
            "train",
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            lvc_chain=self._lvc_chain,
            shuffle=True,
        )
        if self._num_batches is None:
            self._num_batches = self._model.__dict__.get("num_batches")

        return probe(
            self._model,
            self._dataloader,
            probe_config["dct"],
            probe_config["idct"],
            nn.Identity(),
            self._criterion,
            preprocess=probe_config["dct_preprocess"],
            postprocess=probe_config["idct_postprocess"],
            device=self._device,
            subtract_mean=probe_config["subtract_mean"],
            num_batches=self._num_batches,
            do_print=self._do_print,
        )

    @property
    def img_w(self) -> int:
        return 2048

    @property
    def img_h(self) -> int:
        return 1024
