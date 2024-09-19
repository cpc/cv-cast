import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from PIL.Image import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .local_dataset import LocalDataset

CITYSCAPES_IGNORE_LABEL = 255
CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_MEAN = [0.485, 0.456, 0.406]
CITYSCAPES_STD = [0.229, 0.224, 0.225]


def pil_to_tensor_int(pil_image: Image):
    return torch.as_tensor(np.array(pil_image), dtype=torch.int64)


def get_cityscapes_dataloader(
    root: str | Path,
    split: Literal["train", "val"],
    batch_size: int = 1,
    num_workers: int = 0,
    lvc_chain: transforms.Compose | None = None,
    shuffle: bool = False,
) -> DataLoader:
    if lvc_chain is None:
        transform_parts = [
            transforms.ToTensor(),
            # transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD),
        ]
    else:
        transform_parts = [
            transforms.ToTensor(),
            lvc_chain,
            # transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD),
        ]

    dataset = datasets.Cityscapes(
        root,
        split=split,
        mode="fine",
        target_type="semantic",
        transform=transforms.Compose(transform_parts),
        target_transform=pil_to_tensor_int,
        transforms=None,
    )

    if shuffle:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(0)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

def get_folder_dataloader(
    root: str | Path,
    batch_size: int = 1,
    num_workers: int = 0,
    lvc_chain: transforms.Compose | None = None,
    shuffle: bool = False,
) -> DataLoader:
    if lvc_chain is None:
        transform_parts = [
            transforms.ToTensor(),
            # transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD),
        ]
    else:
        transform_parts = [
            transforms.ToTensor(),
            lvc_chain,
            # transforms.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD),
        ]

    dataset = datasets.ImageFolder(
        root,
        transform=transforms.Compose(transform_parts),
        target_transform=pil_to_tensor_int,
    )

    if shuffle:

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(0)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def eval_psnr(dataloader: DataLoader, ref_dataloader: DataLoader) -> float:
    for batch, ref_batch in zip(dataloader, ref_dataloader):
        print(batch.min(), batch.max())
