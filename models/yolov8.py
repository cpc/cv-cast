import os
from pathlib import Path
from typing import Tuple, List, Literal

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .model import Model, ModelConfig


class YoloConfig(ModelConfig):
    task: str
    data_file: str | Path  # coco.yaml, coco128.yaml


class YoloModel(Model):
    def __init__(
        self,
        config: YoloConfig,
        device: str | torch.device | None = None,
        num_batches: int | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        lvc_chain: transforms.Compose | None = None,
        color_space: str = "yuv",
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

        os.environ["DO_PRINT"] = str(do_print)

        from ultralytics import YOLO

        self._task = config["task"]
        self._data_file = config["data_file"]
        self._model = YOLO(self._snapshot, self._task, do_print=do_print)

    def _get_dataloader(self, split: Literal["train", "val"]) -> DataLoader:
        if split == "val":
            return self._model.get_val_dataloader(
                lvc_chain=self._lvc_chain,
                num_batches=self._num_batches,
                do_print=self._do_print,
                batch=self._batch_size,
                workers=self._num_workers,
                data=self._data_file,
            )
        elif split == "train":
            raise ValueError(f"Unsupported dataloader split: {split}")
        else:
            raise ValueError(f"Unknown dataloader split: {split}")

    def _get_batch_images(self, batch) -> torch.Tensor:
        return batch["img"].to(self._device, non_blocking=True) / 255.0

    def _probe(self, probe_config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_dct, mean_dct_grad = self._model.probe(
            probe_config,
            lvc_chain=self._lvc_chain,
            do_print=self._do_print,
            data=self._data_file,
            resume=False,  # if True, resets params like batch size etc.
            pretrained=True,
            device=str(self._device),
            batch=self._batch_size,
            workers=self._num_workers,
            seed=probe_config["seed"],
            amp=False,  # Enabling automatic mixed precision causes NaNs in gradients
        )
        if self._num_batches is None:
            self._num_batches = self._model.__dict__.get("num_batches")

        return mean_dct, mean_dct_grad

    def _get_params(self) -> List:
        return list(self._model.model.parameters())

    def eval(self) -> dict:
        res = self._model.val(
            lvc_chain=self._lvc_chain,
            num_batches=self._num_batches,
            do_print=self._do_print,
            batch=self._batch_size,
            workers=self._num_workers,
            data=self._data_file,
        )
        if self._num_batches is None:
            self._num_batches = self._model.__dict__.get("num_batches")

        return {
            "mAP_50_95": res.json_res["metrics/mAP50-95(B)"],
            "mAP_50_95_partial": res.results_dict["metrics/mAP50-95(B)"],
        }

    @property
    def img_w(self) -> int:
        return 640

    @property
    def img_h(self) -> int:
        return 640
