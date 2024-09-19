from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LocalDataset(Dataset):
    def __init__(
        self, image_names: list, labels: list[int], transform: transforms.Compose
    ):
        def is_file_ok(fp):
            return fp.exists() and fp.is_file()

        for image_name in image_names:
            if not is_file_ok(Path(image_name)):
                raise ValueError("Could not open file:", image_name)

        if len(labels) != len(image_names):
            raise ValueError(
                "The lengths of image names ({}) and labels ({}) do not match".format(
                    len(image_names), len(labels)
                )
            )

        self.image_names = []
        self.labels = []
        for image_name, label in zip(image_names, labels):
            img = Image.open(image_name).convert(mode="RGB")
            if img.mode == "RGB":
                self.image_names.append(image_name)
                self.labels.append(label)
            else:
                print(
                    "Warning: Image mode must be 'RGB', got '{}' for image '{}'".format(
                        img.mode, image_name
                    )
                )

        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_names[i]).convert(mode="RGB")

        image_tensor = self.transform(image)  # there is always a transform

        return (image_tensor, self.labels[i])
