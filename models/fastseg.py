# butchered code from semantic-segmentation (fastseg)

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
from torchvision.transforms.functional import to_pil_image

from .model import Model, ModelConfig, plot_channels
from datasets import (
    CITYSCAPES_IGNORE_LABEL,
    CITYSCAPES_NUM_CLASSES,
    CITYSCAPES_MEAN,
    CITYSCAPES_STD,
    get_cityscapes_dataloader,
    get_folder_dataloader,
    cityscapes_labels,
)
from utils import block_process
from transforms.dct import Dct, Idct
from transforms.color_transforms import RgbToYcbcr, YcbcrToRgb
from utils import set_device

DATASET_IGNORE_LABEL = CITYSCAPES_IGNORE_LABEL
DATASET_NUM_CLASSES = CITYSCAPES_NUM_CLASSES
DATASET_MEAN = CITYSCAPES_MEAN
DATASET_STD = CITYSCAPES_STD


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(
        self,
        weight=None,
        ignore_index=DATASET_IGNORE_LABEL,
        reduction="mean",
        do_print: bool = False,
    ):
        super(CrossEntropyLoss2d, self).__init__()
        if do_print:
            print("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss(
            weight, reduction=reduction, ignore_index=ignore_index
        )

    def forward(self, inputs, targets, do_rmi=None):
        torch.use_deterministic_algorithms(
            False
        )  # work around non-deterministic NLLLoss
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class MobileNetV3LargeTrunk(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_large_100, pretrained=False):
        super(MobileNetV3LargeTrunk, self).__init__()
        net = trunk(pretrained=pretrained, norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[3][0].conv_dw.stride = (1, 1)
        net.blocks[5][0].conv_dw.stride = (1, 1)

        for block_num in (3, 4, 5, 6):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 5:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(
                    m, Conv2dSameExport
                ):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]

    def forward(self, x):
        x = self.early(x)  # 2x
        x = self.block0(x)
        s2 = x
        x = self.block1(x)  # 4x
        s4 = x
        x = self.block2(x)  # 8x
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return s2, s4, x


class MobileNetV3SmallTrunk(nn.Module):
    def __init__(self, trunk=tf_mobilenetv3_small_100, pretrained=False):
        super(MobileNetV3SmallTrunk, self).__init__()
        net = trunk(pretrained=pretrained, norm_layer=nn.BatchNorm2d)

        self.early = nn.Sequential(net.conv_stem, net.bn1, net.act1)

        net.blocks[2][0].conv_dw.stride = (1, 1)
        net.blocks[4][0].conv_dw.stride = (1, 1)

        for block_num in (2, 3, 4, 5):
            for sub_block in range(len(net.blocks[block_num])):
                sb = net.blocks[block_num][sub_block]
                if isinstance(sb, InvertedResidual):
                    m = sb.conv_dw
                else:
                    m = sb.conv
                if block_num < 4:
                    m.dilation = (2, 2)
                    pad = 2
                else:
                    m.dilation = (4, 4)
                    pad = 4
                # Adjust padding if necessary, but NOT for "same" layers
                assert m.kernel_size[0] == m.kernel_size[1]
                if not isinstance(m, Conv2dSame) and not isinstance(
                    m, Conv2dSameExport
                ):
                    pad *= (m.kernel_size[0] - 1) // 2
                    m.padding = (pad, pad)

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]

    def forward(self, x):
        x = self.early(x)  # 2x
        s2 = x
        x = self.block0(x)  # 4x
        s4 = x
        x = self.block1(x)  # 8x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return s2, s4, x


def get_trunk(trunk_name, output_stride=8, do_print: bool = False):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, "Only stride8 supported right now"

    if trunk_name == "mobilenetv3_large":
        backbone = MobileNetV3LargeTrunk(pretrained=False)
        s2_ch = 16
        s4_ch = 24
        high_level_ch = 960
    elif trunk_name == "mobilenetv3_small":
        backbone = MobileNetV3SmallTrunk(pretrained=False)
        s2_ch = 16
        s4_ch = 16
        high_level_ch = 576
    else:
        raise "unknown backbone {}".format(trunk_name)

    if do_print:
        print("Trunk: {}".format(trunk_name))
    return backbone, s2_ch, s4_ch, high_level_ch


def Norm2d(in_channels, **kwargs):
    """
    Custom Norm Function to allow flexible switching
    """
    # layer = getattr(cfg.MODEL, 'BNFUNC')
    # normalization_layer = layer(in_channels, **kwargs)
    # return normalization_layer
    return nn.BatchNorm2d


class ConvBnRelu(nn.Module):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class LRASPP(nn.Module):
    """Lite R-ASPP style segmentation network."""

    def __init__(
        self,
        num_classes,
        trunk,
        criterion=None,
        use_aspp=False,
        num_filters=128,
        do_print: bool = False,
    ):
        """Initialize a new segmentation model.

        Keyword arguments:
        num_classes -- number of output classes (e.g., 19 for Cityscapes)
        trunk -- the name of the trunk to use ('mobilenetv3_large', 'mobilenetv3_small')
        use_aspp -- whether to use DeepLabV3+ style ASPP (True) or Lite R-ASPP (False)
            (setting this to True may yield better results, at the cost of latency)
        num_filters -- the number of filters in the segmentation head
        """
        super(LRASPP, self).__init__()

        self.criterion = criterion
        self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(
            trunk_name=trunk, do_print=do_print
        )
        self.use_aspp = use_aspp

        # Reduced atrous spatial pyramid pooling
        if self.use_aspp:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=12, padding=12),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv3 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=36, padding=36),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            aspp_out_ch = num_filters * 4
        else:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Sigmoid(),
            )
            aspp_out_ch = num_filters

        self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(aspp_out_ch, num_filters, kernel_size=1)
        self.conv_up2 = ConvBnRelu(num_filters + 64, num_filters, kernel_size=1)
        self.conv_up3 = ConvBnRelu(num_filters + 32, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, inputs, gts=None):
        assert "images" in inputs
        x = inputs["images"]

        s2, s4, final = self.trunk(x)
        if self.use_aspp:
            aspp = torch.cat(
                [
                    self.aspp_conv1(final),
                    self.aspp_conv2(final),
                    self.aspp_conv3(final),
                    F.interpolate(self.aspp_pool(final), size=final.shape[2:]),
                ],
                1,
            )
        else:
            aspp = self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode="bilinear", align_corners=False)

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode="bilinear", align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)

        if self.training:
            assert "gts" in inputs
            gts = inputs["gts"]
            return self.criterion(y, gts)
        return {"pred": y}


class MobileV3Large(LRASPP):
    """MobileNetV3-Large segmentation network."""

    model_name = "mobilev3large-lraspp"

    def __init__(self, num_classes, criterion, **kwargs):
        super(MobileV3Large, self).__init__(
            num_classes, criterion=criterion, trunk="mobilenetv3_large", **kwargs
        )


class MobileV3Small(LRASPP):
    """MobileNetV3-Small segmentation network."""

    model_name = "mobilev3small-lraspp"

    def __init__(self, num_classes, criterion, **kwargs):
        super(MobileV3Small, self).__init__(
            num_classes, criterion=criterion, trunk="mobilenetv3_small", **kwargs
        )


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


CITYSCAPES_PALETTE = [
    128,
    64,
    128,
    244,
    35,
    232,
    70,
    70,
    70,
    102,
    102,
    156,
    190,
    153,
    153,
    153,
    153,
    153,
    250,
    170,
    30,
    220,
    220,
    0,
    107,
    142,
    35,
    152,
    251,
    152,
    70,
    130,
    180,
    220,
    20,
    60,
    255,
    0,
    0,
    0,
    0,
    142,
    0,
    0,
    70,
    0,
    60,
    100,
    0,
    80,
    100,
    0,
    0,
    230,
    119,
    11,
    32,
]


# https://github.com/ekzhang/fastseg/blob/master/infer.py
def colorize(mask_array: ArrayLike):
    """Colorize a segmentation mask.

    Keyword arguments:
    mask_array -- the segmentation as a 2D numpy array of integers [0..classes - 1]
    palette -- the palette to use (default 'cityscapes')
    """
    mask_img = Image.fromarray(mask_array.astype(np.uint8)).convert("P")
    mask_img.putpalette(CITYSCAPES_PALETTE)
    return mask_img.convert("RGB")


def blend(input_img, seg_img):
    """Blend an input image with its colorized segmentation labels."""
    return Image.blend(input_img, seg_img, 0.4)


class FastsegConfig(ModelConfig):
    pass


class FastsegModel(Model):
    def __init__(
        self,
        config: FastsegConfig,
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

        self._variant = config["variant"]
        self._criterion = CrossEntropyLoss2d(do_print=do_print)
        self._cityscapes_root = Path.home() / "data/cityscapes"

        if self._variant == "small":
            self._model = MobileV3Small(
                DATASET_NUM_CLASSES, self._criterion, do_print=do_print
            )
        elif self._variant == "large":
            self._model = MobileV3Large(
                DATASET_NUM_CLASSES, self._criterion, do_print=do_print
            )
        else:
            raise ValueError(f"Unknown FastSeg variant: {self._variant}")

        checkpoint = torch.load(self._snapshot, map_location=self._device)
        if do_print:
            print("checkpoint: ", list(checkpoint.keys()))

        restore_net(self._model, checkpoint)
        self._model = self._model.to(device)

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

    def _get_params(self) -> list:
        return list(self._model.parameters())

    def predict(self, custom_dir: Path | None) -> Tuple[List, List, List, List]:
        if custom_dir is None:
            self._dataloader = get_cityscapes_dataloader(
                self._cityscapes_root,
                "val",
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                lvc_chain=self._lvc_chain,
                shuffle=True,
            )
        else:
            self._dataloader = get_folder_dataloader(
                custom_dir,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                lvc_chain=self._lvc_chain,
                shuffle=True,
            )
        if self._num_batches is None:
            self._num_batches = self._model.__dict__.get("num_batches")

        res = main_eval(
            self._model,
            self._criterion,
            self._dataloader,
            self._device,
            num_batches=self._num_batches,
            do_print=self._do_print,
            do_predict=True,
        )

        collected_images = res["collected"]

        to_pil_img = T.ToPILImage()

        res_colorized = []
        res_blended = []
        res_orig = []
        res_gt = []

        for collected in collected_images:
            inp = collected["inp"]
            out = collected["out"]
            gt = collected["gt"]

            colorized = colorize(out.cpu().numpy())
            res_colorized.append(colorized)
            res_blended.append(blend(to_pil_img(inp), colorized))
            res_orig.append(to_pil_img(inp))
            try:
                res_gt.append(colorize(gt.cpu().numpy()))
            except IndexError:
                res_gt.append(colorize(np.zeros_like(out.cpu().numpy())))

        return (res_colorized, res_blended, res_orig, res_gt)

    def eval(self) -> dict:
        self._dataloader = get_cityscapes_dataloader(
            self._cityscapes_root,
            "val",
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            lvc_chain=self._lvc_chain,
            shuffle=False,
        )
        if self._num_batches is None:
            self._num_batches = self._model.__dict__.get("num_batches")

        return main_eval(
            self._model,
            self._criterion,
            self._dataloader,
            self._device,
            num_batches=self._num_batches,
            do_print=self._do_print,
        )

    @property
    def img_w(self) -> int:
        return 2048

    @property
    def img_h(self) -> int:
        return 1024


def pil_to_tensor_int(pil_image):
    return torch.as_tensor(np.array(pil_image), dtype=torch.int64)


def restore_opt(optimizer, checkpoint):
    assert "optimizer" in checkpoint, "cant find optimizer in checkpoint"
    optimizer.load_state_dict(checkpoint["optimizer"])


def restore_net(net, checkpoint):
    assert "state_dict" in checkpoint, "cant find state_dict in checkpoint"
    forgiving_state_restore(net, checkpoint["state_dict"])


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if new_k not in loaded_dict:
            new_k = "module." + k
        if (
            new_k in loaded_dict
            and net_state_dict[k].size() == loaded_dict[new_k].size()
        ):
            new_loaded_dict[k] = loaded_dict[new_k]
        else:
            print("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net


def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    # stretch ground truth labels by num_classes
    #   class 0  -> 0
    #   class 1  -> 19
    #   class 18 -> 342
    #
    # TP at 0 + 0, 1 + 1, 2 + 2 ...
    #
    # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    hist = np.bincount(
        num_classes * gtruth[mask].astype(int) + pred[mask], minlength=num_classes**2
    )
    hist = hist.reshape(num_classes, num_classes)
    return hist


def calculate_iou(hist_data):
    diag = np.diag(hist_data).astype(np.float64)
    hist_data = hist_data.astype(np.float64)

    if hist_data.sum() == 0:
        acc = 0.0
    else:
        acc = diag.sum() / hist_data.sum()

    hist_sum1 = hist_data.sum(axis=1)
    acc_cls = np.divide(diag, hist_sum1, out=np.zeros_like(diag), where=hist_sum1 != 0)
    acc_cls = np.mean(acc_cls)

    hist_sum0 = hist_data.sum(axis=0)
    divisor = hist_sum1 + hist_sum0 - diag
    iu = np.divide(diag, divisor, out=np.zeros_like(diag), where=divisor != 0)

    return iu, acc, acc_cls


def eval(
    model,
    criterion,
    dataloader,
    device,
    num_batches=None,
    do_print: bool = False,
    do_predict: bool = False,
):
    total_num_batches = len(dataloader)

    if num_batches is None:
        num_batches = total_num_batches
        batch_iter = dataloader
    else:
        batch_iter = itertools.islice(dataloader, num_batches)

    if do_print:
        print(f"num batches: {num_batches} (total {total_num_batches})")

    collected_images = []

    model.eval()
    with torch.inference_mode():
        # output = torch.tensor([0.0]).to(device)
        iou_acc_agg = 0
        val_loss = AverageMeter()
        # losses = []

        normalize = T.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)

        if do_print:
            print("")
        for i, (images, labels) in enumerate(batch_iter):
            if do_print:
                print(
                    f"\rbatch {i+1}/{num_batches}, batch shape: {images.shape}",
                    end="",
                    flush=True,
                )
            batch_pixel_size = images.size(0) * images.size(2) * images.size(3)

            labels = (
                torch.tensor(cityscapes_labels.id2train_id)
                .take(labels)
                .to(device, non_blocking=True)
            )

            images = images.to(device)
            labels = labels.to(device)
            images_orig = images

            if do_print:
                for j, img in enumerate(images):
                    if torch.all(img < 1e-5):
                        print(f"WARNING: Likely got black image no. {j}, batch {i}")

            images = normalize(images)

            # torch.autograd.set_detect_anomaly(True)
            inp = {
                "images": images,
                "gts": labels,
            }
            out_dict = model(inp)
            pred = out_dict["pred"]

            if do_predict:
                for ii, oo, lab in zip(images_orig, pred.argmax(dim=1), labels):
                    collected_images.append({"inp": ii, "out": oo, "gt": lab})
            else:
                loss = criterion(pred, labels)
                val_loss.update(loss, batch_pixel_size)

                output_data = F.softmax(pred, dim=1).data
                max_probs, predictions = output_data.max(1)

                iou_acc = fast_hist(
                    predictions.cpu().numpy().flatten(),
                    labels.cpu().numpy().flatten(),
                    DATASET_NUM_CLASSES,
                )
                iou_acc_agg += iou_acc

        if do_print:
            print("")

    return val_loss, iou_acc_agg, collected_images


def probe(
    model: nn.Module,
    dataloader: DataLoader,
    dct: nn.Module | T.Compose,
    idct: nn.Module | T.Compose,
    activation: nn.Module,
    loss: nn.Module,
    preprocess: nn.ModuleList | None = None,
    postprocess: nn.ModuleList | None = None,
    device: torch.device | str = "cpu",
    subtract_mean: bool = True,
    num_batches: int | None = None,
    do_print: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    total_num_batches = len(dataloader)

    if num_batches is None:
        num_batches = total_num_batches
        batch_iter = dataloader
    else:
        batch_iter = itertools.islice(dataloader, num_batches)

    if do_print:
        print(f"num batches: {num_batches} (total {total_num_batches})")
        print("")

    model.eval()
    res_dct = []
    res_dct_grad = []

    normalize = T.Normalize(CITYSCAPES_MEAN, CITYSCAPES_STD)

    for batch_id, (images, batch_labels) in enumerate(batch_iter):
        if do_print:
            print(
                f"\rbatch {batch_id+1}/{num_batches}: data {images.shape}, labels {batch_labels.shape}",
                end="",
                flush=True,
            )

        images: torch.Tensor = images.to(device, non_blocking=True)

        # Uncomment to save the input images for reference:
        # for i, img in enumerate(images):
        #     fname = f"batch{batch_id}_img{i}.png"
        #     T.ToPILImage()(img).save(fname)

        # images.requires_grad_(True)
        # images.retain_grad()
        batch_labels = (
            torch.tensor(cityscapes_labels.id2train_id)
            .take(batch_labels)
            .to(device, non_blocking=True)
        )
        batch_labels = batch_labels.to(device, non_blocking=True)

        if preprocess is None:
            inp_images = images
        else:
            metas = []
            imgs = []
            for im in images:
                for layer in preprocess:
                    if type(im) == tuple:
                        im = layer(*im)
                    else:
                        im = layer(im)
                if type(im) == tuple:
                    im, meta = im
                else:
                    meta = None
                metas.append(meta)
                imgs.append(im)
            inp_images = torch.stack(imgs)

        if subtract_mean:
            means = inp_images.mean(dim=(2, 3))
            inp_images = inp_images - means[:, :, None, None]

        xdct = dct(inp_images)
        xdct.requires_grad_(True)
        xdct.retain_grad()

        xidct = idct(xdct)

        if subtract_mean:
            xidct = xidct + means[:, :, None, None]

        if postprocess is None:
            out_images = xidct
        else:
            imgs = []
            for im in zip(xidct, metas):
                for layer in postprocess:
                    if type(im) == tuple:
                        im, meta = im
                        if meta is None:
                            im = layer(im)
                        else:
                            im = layer(im, meta)
                    else:
                        im = layer(im)
                imgs.append(im)
            out_images = torch.stack(imgs)

        out_images = normalize(out_images)

        inp = {
            "images": out_images,
            "gts": batch_labels,
        }

        # torch.autograd.set_detect_anomaly(True) # <-- DEBUG
        out = model(inp)["pred"]
        output = activation(out)

        loss_value = loss(output, batch_labels)

        # Calculate images.grad = dloss/dimages for every images with images.requires_grad=True
        model.zero_grad()
        loss_value.backward()

        # mean_dct_grad = mean_dct_grad + xdct.grad.abs().mean(dim=0).detach().clone()
        # mean_dct = mean_dct + xdct.mean(dim=0).detach().clone()
        res_dct_grad.append(xdct.grad.detach().clone())
        res_dct.append(xdct.detach().clone())

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

    if do_print:
        print("")

    return (torch.cat(res_dct), torch.cat(res_dct_grad))


def main_eval(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    num_batches: int | None = None,
    do_print: bool = False,
    do_predict: bool = False,
) -> dict:
    val_loss, iou_acc, collected_images = eval(
        model,
        criterion,
        dataloader,
        device,
        num_batches=num_batches,
        do_print=do_print,
        do_predict=do_predict,
    )

    if do_predict:
        iu, acc, acc_cls = 0.0, 0.0, 0.0
        avg_loss = val_loss.avg
    else:
        iu, acc, acc_cls = calculate_iou(iou_acc)
        avg_loss = val_loss.avg.item()

    mean_iu = np.nanmean(iu)
    if do_print:
        if do_predict:
            print(f"Mean IoU: N/A")
            print(f"Val loss: N/A")
        else:
            print(f"Mean IoU: {mean_iu}")
            print(f"Val loss: {val_loss.avg}")

    return {
        "mean_iu": mean_iu,
        "val_loss_avg": avg_loss,
        "collected": collected_images,
    }


def main_probe(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    color_space: str = "rgb",
    num_batches: int | None = None,
    show_plots: bool = False,
    do_print: bool = False,
) -> dict:
    if show_plots:
        plt.rcParams["figure.dpi"] = 200
        plt.rcParams["savefig.dpi"] = 200

    _, grads = probe(
        model,
        dataloader,
        Dct(),
        Idct(),
        nn.Identity(),
        criterion,
        device=device,
        subtract_mean=False,
        num_batches=num_batches,
        do_print=do_print,
    )

    if do_print:
        print("gradients: ", grads.shape)

    # grads_norm = grads.abs().mean(dim=0)
    # grads_norm = grads.square().sum(dim=0).sqrt()
    # grads_norm = grads.square()

    # sens_rgb = grads_norm.sum(dim=(1, 2))
    sens_rgb = grads.square().sum(dim=(1, 2)).sqrt()
    sens_rgb = sens_rgb.div(sens_rgb.sum())
    if do_print:
        print("\nSensitivity to RGB channels:", sens_rgb, ", sum:", sens_rgb.sum())

    # Compute YUV weights
    z1 = torch.median(grads[2] / grads[1])
    z2 = torch.median(grads[0] / grads[1])
    Wr = z2 / (1 + z1 + z2)
    Wg = 1 / (1 + z1 + z2)
    Wb = z1 / (1 + z1 + z2)
    W = torch.Tensor((Wr, Wg, Wb))
    W_human = torch.Tensor((0.299, 0.587, 0.114))

    # Human values (CCIR.601):
    # Wr = 0.299
    # Wg = 0.587
    # Wb = 0.114
    if do_print:
        print(
            "\nWr: {:.4f}, Wg: {:.4f}, Wb: {:.4f}, sum: {}".format(
                Wr, Wg, Wb, Wr + Wg + Wb
            )
        )
        print("human:", W_human)

    if show_plots:
        plot_channels(
            grads.abs().cpu().numpy(),
            [
                "abs DCT gradient (R)",
                "abs DCT gradient (G)",
                "abs DCT gradient (B)",
            ],
            0,
            show=False,
        )

    # Calculate per-chunk norm of the YUV sensitivity map
    chunk_size = (128, 256)

    if (grads.shape[2] % chunk_size[1] != 0) or (grads.shape[1] % chunk_size[0] != 0):
        print("Image size not divisible by chunk size")
        if show_plots:
            plt.show()
        sys.exit(1)

    if do_print:
        print("\nCalculating per-chunk norm of YUV sensitivity")

    norm = lambda tensor: tensor.square().sum().sqrt()
    grads_norm = block_process(grads, chunk_size, norm)

    if show_plots:
        plot_channels(
            grads_norm.cpu().numpy(),
            [
                "block_norm DCT gradient (R)",
                "block_norm DCT gradient (G)",
                "block_norm DCT gradient (B)",
            ],
            1,
            show=False,
        )

    if color_space == "yuv":
        preprocess = nn.ModuleList([RgbToYcbcr(W)])
        postprocess = nn.ModuleList([YcbcrToRgb(W)])

        _, grads_yuv = probe(
            model,
            dataloader,
            Dct(),
            Idct(),
            nn.Identity(),
            criterion,
            preprocess=preprocess,
            postprocess=postprocess,
            device=device,
            subtract_mean=False,
            num_batches=num_batches,
            do_print=do_print,
        )

        sens_yuv = grads_yuv.square().sum(dim=(1, 2)).sqrt()
        sens_yuv = sens_yuv.div(sens_yuv.sum())
        if do_print:
            print("\nSensitivity to YUV channels:", sens_yuv, ", sum:", sens_yuv.sum())

        if show_plots:
            plot_channels(
                grads_yuv.abs().cpu().numpy(),
                [
                    "abs DCT gradient (Y)",
                    "abs DCT gradient (U)",
                    "abs DCT gradient (V)",
                ],
                2,
                show=False,
            )

        grads_norm = block_process(grads_yuv, chunk_size, norm)

        if show_plots:
            plot_channels(
                grads_norm.cpu().numpy(),
                [
                    "block_norm DCT gradient (Y)",
                    "block_norm DCT gradient (U)",
                    "block_norm DCT gradient (V)",
                ],
                3,
                show=True,
            )

    return {
        "grads_norm": grads_norm,
        "W": W,
    }


def main(
    mode: str = "eval",
    device: str | torch.device | None = None,
    num_batches: int | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    lvc_chain: T.Compose | None = None,
    color_space: str = "rgb",
    do_print: bool = False,
    show_plots: bool = False,
    variant="large",
) -> dict:
    if color_space not in ["yuv", "rgb"]:
        raise ValueError(f"Unknown color space: {color_space}")

    if not device:
        device = set_device(do_print=do_print)

    criterion = CrossEntropyLoss2d(do_print=do_print)
    net_small = MobileV3Small(DATASET_NUM_CLASSES, criterion, do_print=do_print)
    net_large = MobileV3Large(DATASET_NUM_CLASSES, criterion, do_print=do_print)

    # cityscapes_root = "/home/kubouch/data/cityscapes"
    # cityscapes_root = "/mnt/1tb_storage/data/cityscapes"
    cityscapes_root = Path.home() / "data/cityscapes"

    split = "val" if mode == "eval" or mode == "eval_lvc" else "train"

    dataloader = get_cityscapes_dataloader(
        cityscapes_root,
        split,
        batch_size=batch_size,
        num_workers=num_workers,
        lvc_chain=lvc_chain,
        shuffle=mode == "probe",
    )

    if variant == "large":
        model = net_large
        snapshot = (
            Path.home() / "data/models/fastseg/raw/large/best_checkpoint_ep172.pth"
        )

    else:
        model = net_small
        snapshot = (
            Path.home() / "data/models/fastseg/raw/small/best_checkpoint_ep171.pth"
        )

    if "ep167" in str(snapshot):
        print("=======================================")
        print(f"WARNING! Likely wrong model snapshot: {snapshot}")
        print("=======================================")
    checkpoint = torch.load(snapshot, map_location=device)
    if do_print:
        print("checkpoint: ", list(checkpoint.keys()))
    restore_net(model, checkpoint)
    model = model.to(device)

    if (mode == "eval") or (mode == "eval_lvc"):
        return main_eval(
            model,
            criterion,
            dataloader,
            device,
            num_batches=num_batches,
            do_print=do_print,
        )
    elif mode == "probe":
        return main_probe(
            model,
            criterion,
            dataloader,
            device,
            color_space=color_space,
            num_batches=num_batches,
            show_plots=show_plots,
            do_print=do_print,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
