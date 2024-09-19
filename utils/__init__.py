import torch
from typing import Callable, NamedTuple, Tuple

from torch import Tensor


class Size(NamedTuple):
    """2D size tuple in PyTorch order (H, W)"""
    h: int  # height
    w: int  # width


def set_device(device=None, do_print: bool = True) -> torch.device:
    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)

    if do_print:
        print("Using device:", selected_device)

    if selected_device.type == "cuda" and do_print:
        print(torch.cuda.get_device_name(selected_device.index))
        print("Current device: ", torch.cuda.current_device())
        print(" - available:", torch.cuda.is_available())
        print("Total device count: ", torch.cuda.device_count())

    return selected_device


def block_process(inp: Tensor, block_size: Tuple[int, int], fn: Callable) -> Tensor:
    """Process input 3-dim tensor block-by-block, mapping each block into a single
    value"""

    nch, h, w = inp.shape
    bh, bw = block_size

    if (w % bw != 0) or (h % bh != 0):
        raise ValueError(f"Image size {w}x{h} not divisible by block size {bw}x{bh}.")

    nb_x = w // bw
    nb_y = h // bh

    res = torch.zeros(nch, nb_y, nb_x)

    for ch in range(nch):
        for by in range(nb_y):
            for bx in range(nb_x):
                y = by * bh
                x = bx * bw
                y1 = y + bh
                x1 = x + bw

                res[ch, by, bx] = fn(inp[ch, y:y1, x:x1])

    return res


def block_process_exp(inp: Tensor, block_size: Tuple[int, int], fn: Callable) -> Tensor:
    """Process input 3-dim tensor block-by-block, mapping each block into a block of
    the same size (i.e., preserving the input shape)"""

    nch, h, w = inp.shape
    bh, bw = block_size

    if (w % bw != 0) or (h % bh != 0):
        raise ValueError(f"Image size {w}x{h} not divisible by block size {bw}x{bh}.")

    nb_x = w // bw
    nb_y = h // bh

    res = torch.zeros_like(inp)

    for ch in range(nch):
        for by in range(nb_y):
            for bx in range(nb_x):
                y = by * bh
                x = bx * bw
                y1 = y + bh
                x1 = x + bw

                res[ch, y:y1, x:x1] = fn(inp[ch, y:y1, x:x1])

    return res


# def get_chunk_size(model_config: ModelConfig, nchunks: int) -> Tuple[int, int]:
#     if not isinstance(nchunks, int):
#         raise NotImplementedError

#     dct_size = math.sqrt(nchunks)

#     if dct_size == int(dct_size):
#         dct_size = int(dct_size)
#     else:
#         raise ValueError(f"Invalid number of chunks: {nchunks}")

#     if model_config["name"] == "yolov8":
#         w = 640
#         h = 640
#     else:
#         w = 2048
#         h = 1024

#     if (w / dct_size != int(w / dct_size)) or (h / dct_size != int(h / dct_size)):
#         raise ValueError(
#             f"Image size {w}x{h} not evenly divisible by DCT size {dct_size}"
#         )

#     return (int(h / dct_size), int(w / dct_size))


def normalize(array, oldmin=0.0, oldmax=255.0, newmin=-1.2, newmax=1.2):
    """Normalize values in array ranging from oldmin..oldmax to a new range
    newmin..newmax
    """

    a = (newmax - newmin) / (oldmax - oldmin)
    b = newmax - a * oldmax
    return a * array + b
