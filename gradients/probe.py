"""Probe neural network with images to acquire gradients
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import LocalDataset


def probe_images(
    model: nn.Module,
    inp_images: list,
    ground_truth,
    preprocess: nn.Module | transforms.Compose,
    dct: nn.Module | transforms.Compose,
    idct: nn.Module | transforms.Compose,
    activation: nn.Module,
    loss: nn.Module,
    device="cpu",
    batch_size: int = 32,
    subtract_mean: bool = True,
) -> torch.Tensor:
    dataset = LocalDataset(inp_images, labels=ground_truth, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    return probe(
        model,
        dataloader,
        dct,
        idct,
        activation,
        loss,
        device=device,
        subtract_mean=subtract_mean,
    )


def probe(
    model: nn.Module,
    dataloader: DataLoader,
    dct: nn.Module | transforms.Compose,
    idct: nn.Module | transforms.Compose,
    activation: nn.Module,
    loss: nn.Module,
    device="cpu",
    subtract_mean: bool = True,
) -> torch.Tensor:
    """Probe neural network with images to acquire gradients

    Inputs:
        model: The neural network to probe
        dataloader: DataLoader instance for iterating minibatches
        dct: Discrete cosine transform module
        idct: Transforms to apply to the DCT coefficients
        activation: Activation function to apply to the last layer
        loss: Loss function
        (default = 'cpu') device: Which device to use
        (default = True) subtract_mean: Subtract mean value from each channel before the DCT

    Returns:
        Tensor of gradients w.r.t. to DCT coefficients for each image
    """

    model.to(device)

    dct_gradients = []

    for batch_id, (images, batch_labels) in enumerate(dataloader):
        print(
            "batch {}: data {}, labels {}".format(
                batch_id, images.shape, batch_labels.shape
            )
        )

        images: torch.Tensor = images.to(device, non_blocking=True)
        if subtract_mean:
            means = images.mean(dim=(2, 3))
            images = images - means[:, :, None, None]
        images.requires_grad_(True)
        images.retain_grad()
        batch_labels = batch_labels.to(device, non_blocking=True)

        xdct = dct(images)

        xdct.requires_grad_(True)
        xdct.retain_grad()

        xidct = idct(xdct)

        if subtract_mean:
            xidct = xidct + means[:, :, None, None]

        # DEBUG
        #  torch.autograd.set_detect_anomaly(True)
        out = model(xidct)
        output = activation(out)

        loss_value = loss(output, batch_labels)

        # Calculate images.grad = dloss/dimages for every images with images.requires_grad=True
        loss_value.backward()

        dct_gradients.append(xdct.grad)

        # if batch_id == show_id[0]:
        #     top_scores = 5
        #     scores, idxs = output[show_id[1]].sort(descending=True)[:top_scores]
        #     for score, idx in zip(scores[:top_scores], idxs[:top_scores]):
        #         idx = int(idx.detach().cpu().numpy())
        #         score = torch.exp(score)
        #         print(
        #             " - score: {:.4f}  idx: {:3d}  label: {}".format(
        #                 score, idx, IDX_TO_LABEL[idx]
        #             )
        #         )

    return torch.cat(dct_gradients)
