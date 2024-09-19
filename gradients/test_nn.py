#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch import Tensor
from torch.nn import NLLLoss, LogSoftmax
from torch.utils.data import Dataset
from PIL import Image
from torchsummary import summary

from transforms.dct import Dct, Idct
from transforms.color_transforms import YcbcrToRgb


# import hack to make this work with fastseg
def versiontuple(v):
    return tuple(map(int, (v.split("."))))


if versiontuple(sys.version.split()[0]) < versiontuple("3.9.0"):
    from ..datasets.imagenet1000_class_to_label import IDX_TO_LABEL, LABEL_TO_IDX
    from ..models.drn import drn_d_22, drn_d_38
    from ..transforms.color_transforms import RgbToYcbcr
    from .probe import probe_images
else:
    from datasets.imagenet1000_class_to_label import IDX_TO_LABEL, LABEL_TO_IDX
    from models.drn import drn_d_22, drn_d_38
    from transforms.color_transforms import RgbToYcbcr
    from gradients.probe import probe_images

# Notes:
# - JPEG uses DCT type 2 and IDCT type 2 (tha latter equals to DCT type 3)
# - 12x8 image is read into tensor with size [3, 8, 12]

WEIGHTS = {
    "alexnet": "AlexNet_Weights.IMAGENET1K_V1",
    "shufflenet_v2_x1_0": "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1",
    "squeezenet1_1": "SqueezeNet1_1_Weights.IMAGENET1K_V1",
    "resnet50": "ResNet50_Weights.IMAGENET1K_V2",
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "vgg11": "VGG11_BN_Weights.IMAGENET1K_V1",
    "drn_d_22": None,
    "drn_d_38": None,
}


class LocalDataset(Dataset):
    def __init__(self, image_names, labels=None, transform=None):
        def is_file_ok(fp):
            return fp.exists() and fp.is_file()

        for image_name in image_names:
            if not is_file_ok(Path(image_name)):
                raise ValueError("Could not open file:", image_name)

        if transform is None:
            transform = transform.Compose([transform.ToTensor()])

        if labels is not None:
            if type(labels) != Tensor:
                labels = torch.tensor(labels, dtype=int)

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

    def __getitem__(self, i):
        image = Image.open(self.image_names[i]).convert(mode="RGB")

        image_tensor = self.transform(image)  # there is always a transform

        if self.labels is not None:
            return (image_tensor, self.labels[i])
        else:
            return image_tensor


def get_imagenet_labels(images: List[str] | List[Path]) -> List[int]:
    res = []

    for image in images:
        label = Path(image).parent.name
        i = None
        for k in LABEL_TO_IDX.keys():
            if label.lower() in k.lower():
                i = LABEL_TO_IDX[k]

        if i is None:
            i = 0

        res.append(i)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inp_dir", type=str, help="Directory with images to load.")
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*.png",
        help="Glob to match images in the inp_dir.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="alexnet",
        help="Which model to use.",
        choices=WEIGHTS.keys(),
    )
    parser.add_argument("-n", type=int, default=10, help="Number of pictures to use")
    args = parser.parse_args()

    N = args.n
    model_name = args.model

    # TODO CLI arguments:
    device = set_device()
    norm = "ortho"
    show_dct = False
    show_gradients = True
    do_dct_test = False
    img_size = (3, 256, 256)  # size of resized images
    inp_size = (3, 224, 224)  # size of cropped images fed into NN

    # Setup data
    pics = [str(img) for img in Path(args.inp_dir).glob(args.glob)][:N]
    labels = get_imagenet_labels(pics)  # np.zeros(len(pics))
    batch_size = min(len(pics), 32)

    # Optional DCT test
    if do_dct_test:
        test_img = Image.open(pics[0]).convert(mode="RGB")
        test_dct(test_img, show=show_dct, norm=norm)

    # Setup model
    print("Loading model", model_name)

    if model_name == "drn_d_22":
        model = drn_d_22(True)
    elif model_name == "drn_d_38":
        model = drn_d_38(True)
    else:
        model = torch.hub.load(
            "pytorch/vision:v0.13.1", model_name, weights=WEIGHTS[model_name]
        )

    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)
    model.eval()
    summary(model, inp_size)

    # parameters
    lr = 10e-3
    nshow = -1
    show_id = (-1, -1)

    activ = LogSoftmax(dim=1)
    #  optimizer = Adadelta(model.parameters(), lr=lr)
    loss_func = NLLLoss()

    ############################################################################
    # Gradient w.r.t. inputs
    # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709
    print("\nInference for RGB images")

    preprocess = T.Compose(
        [
            T.Resize(img_size[1]),
            T.CenterCrop(inp_size[1]),
            T.ToTensor(),
            #  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    batch_g = probe_images(
        model, pics, labels, preprocess, Dct(), Idct(), activ, loss_func
    )

    # Compute RGB gradient and sensitivity over all batches
    g_mean = batch_g.abs().mean(dim=0)
    sens_rgb = g_mean.sum(dim=(1, 2))
    sens_rgb = sens_rgb.div(sens_rgb.sum())
    print("\nSensitivity to RGB channels:", sens_rgb, ", sum:", sens_rgb.sum())

    # Compute YUV weights
    z1 = torch.median(batch_g[:, 2] / batch_g[:, 1])
    z2 = torch.median(batch_g[:, 0] / batch_g[:, 1])
    Wr = z2 / (1 + z1 + z2)
    Wg = 1 / (1 + z1 + z2)
    Wb = z1 / (1 + z1 + z2)
    W = Tensor((Wr, Wg, Wb))
    W_human = Tensor((0.299, 0.587, 0.114))

    # Human values (CCIR.601):
    # Wr = 0.299
    # Wg = 0.587
    # Wb = 0.114
    print(
        "\nWr: {:.4f}, Wg: {:.4f}, Wb: {:.4f}, sum: {}".format(Wr, Wg, Wb, Wr + Wg + Wb)
    )
    print("human:", W_human)

    if show_gradients:
        gmin = g_mean.min()
        #  g_mean.clip_(gmin, 1e-5)
        gmax = g_mean.max()

        plt.close("all")
        plt.figure(1)
        cmap = cm.jet
        plt.subplot(311)
        plt.imshow(g_mean[0], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (RED)")

        plt.subplot(312)
        plt.imshow(g_mean[1], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (GREEN)")

        plt.subplot(313)
        plt.imshow(g_mean[2], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (BLUE)")

        #  plt.show()

    ############################################################################
    # G-YUV gradient w.r.t. inputs
    # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709
    print("\nInference for YUV images")

    preprocess = T.Compose(
        [
            T.Resize(img_size[1]),
            T.CenterCrop(inp_size[1]),
            T.ToTensor(),
            RgbToYcbcr(W),
            #  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    preprocess_human = T.Compose(
        [
            T.Resize(img_size[1]),
            T.CenterCrop(inp_size[1]),
            T.ToTensor(),
            RgbToYcbcr(W_human),
            #  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    postprocess = T.Compose([Idct(), YcbcrToRgb(W)])
    postprocess_human = T.Compose([Idct(), YcbcrToRgb(W_human)])

    batch_g_yuv = probe_images(
        model, pics, labels, preprocess, Dct(), postprocess, activ, loss_func
    )
    batch_g_yuv_human = probe_images(
        model,
        pics,
        labels,
        preprocess_human,
        Dct(),
        postprocess_human,
        activ,
        loss_func,
    )

    # Compute YUV gradient and sensitivity over all batches
    g_yuv_mean = batch_g_yuv.abs().mean(dim=0)
    g_yuv_mean_human = batch_g_yuv_human.abs().mean(dim=0)
    sens_yuv = g_yuv_mean.sum(dim=(1, 2))
    sens_yuv_human = g_yuv_mean_human.sum(dim=(1, 2))
    sens_yuv = sens_yuv.div(sens_yuv.sum())
    sens_yuv_human = sens_yuv_human.div(sens_yuv_human.sum())
    print("\nSensitivity to YUV channels:", sens_yuv, ", sum:", sens_yuv.sum())
    print(
        "                    (human):", sens_yuv_human, ", sum:", sens_yuv_human.sum()
    )

    if show_gradients:
        gmin = g_yuv_mean.min()
        #  g_yuv_mean.clip_(gmin, 1e-5)
        gmax = g_yuv_mean.max()

        #  plt.close("all")
        cmap = cm.jet
        plt.figure(2)
        plt.subplot(311)
        plt.imshow(g_yuv_mean[0], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (Y)")

        plt.subplot(312)
        plt.imshow(g_yuv_mean[1], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (U)")

        plt.subplot(313)
        plt.imshow(g_yuv_mean[2], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (V)")

        #  plt.show()

    # Calculate per-chunk means of the YUV sensitivity map
    chunk_size = (28, 28)

    if (g_yuv_mean.shape[2] % chunk_size[1] != 0) or (
        g_yuv_mean.shape[1] % chunk_size[0] != 0
    ):
        print("Image size not divisible by chunk size")
        plt.show()
        sys.exit(1)

    print("\nCalculating per-chunk means of YUV sensitivity")

    g_yuv_mean_blocks = grad_block_means(g_yuv_mean, chunk_size)

    if show_gradients:
        gmin = g_yuv_mean_blocks.min()
        #  g_yuv_mean_blocks.clip_(gmin, 1e-5)
        gmax = g_yuv_mean_blocks.max()

        #  plt.close("all")
        cmap = cm.jet
        plt.figure(3)
        plt.subplot(311)
        plt.imshow(g_yuv_mean_blocks[0], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (Y)")

        plt.subplot(312)
        plt.imshow(g_yuv_mean_blocks[1], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (U)")

        plt.subplot(313)
        plt.imshow(g_yuv_mean_blocks[2], cmap=cmap, norm=plt.Normalize(gmin, gmax))
        plt.colorbar(fraction=0.045)
        plt.title("sensitivity map -- mean DCT gradient (V)")

        plt.show()
