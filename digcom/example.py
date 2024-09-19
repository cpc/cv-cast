import io
import os
from pathlib import Path
from typing import Union, Tuple

from digcommpy import encoders, decoders, channels, modulators, demodulators, metrics
from imageio.v2 import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL


def unpack_bytes_to_bits(messages, num_bits: int) -> npt.NDArray:
    """Converts an array of bytes into array binary messages of size num_bits.

    Parameters
    ----------
    messages : array (N x 1)
        List of messages (bytes)

    num_bits : int
        Number of output bits

    Returns
    -------
    binary_messages : array (N * 8 / num_bits x num_bits)
        Converted messages as bits
    """
    num_bits = int(num_bits)

    if (len(messages) * 8) % num_bits != 0:
        raise ValueError(
            f"Array of {len(messages)} bytes cannot be split into {num_bits}-bit messages"
        )

    return np.unpackbits(messages).reshape(-1, num_bits)


def pack_bits_to_bytes(messages) -> npt.NDArray:
    """Converts an array of binary numbers into bytes.

    Parameters
    ----------
    messages : array (N * 8 / num_bits x num_bits)
        Array where each row contains one message and each column one bit

    Returns
    -------
    dec_messages : array (N x 1)
        Converted messages as bytes
    """
    if np.product(messages.shape) % 8 != 0:
        raise ValueError(
            f"Array of shape {messages.shape} bits cannot packed into bytes"
        )

    return np.packbits(messages.astype(np.uint8).reshape(-1, 8))


def load_image_raw(image_file: Union[str, Path]) -> Tuple[npt.NDArray, Tuple[int, int]]:
    tmp_pixels = imread(image_file)
    resolution = np.shape(tmp_pixels)
    with open(image_file, "rb") as rf:
        pixels = np.fromfile(rf, dtype=np.uint8)

    return pixels, resolution


def save_image_raw(filepath: Union[str, Path], image: npt.NDArray):
    with open(filepath, "wb") as wf:
        image.astype("uint8").tofile(wf)


# def get_psnr(inp_file, out_file):
#     inp = imread(inp_file)
#     out = imread(out_file)

#     mse = np.square(out.astype(float) - inp.astype(float)).mean()
#     with np.errstate(divide="ignore"):
#         psnr = 10 * np.log10(255**2 / mse)

#     return psnr


def get_psnr(inp_img: npt.NDArray, out_img: npt.NDArray) -> float:
    try:
        out = np.array(PIL.Image.open(io.BytesIO(out_img)))
        inp = np.array(PIL.Image.open(io.BytesIO(inp_img)))

        mse = np.square(out.astype(float) - inp.astype(float)).mean()
        with np.errstate(divide="ignore"):
            psnr = 10 * np.log10(255**2 / mse)

    except OSError:
        # could not decode output
        psnr = 0.0

    return min(psnr, 100)


def transmit_bits(data: np.ndarray, code_length: int, info_bits: int, snr_db: int):
    channel = channels.BawgnChannel(
        snr_db, rate=info_bits / code_length, input_power=1.0
    )
    encoder = encoders.PolarEncoder(code_length, info_bits, channel)
    modulator = modulators.BpskModulator()
    # demodulator = demodulators.BpskDemodulator()
    # modulator = modulators.QamModulator()
    # demodulator = demodulators.QamDemodulator()
    decoder = decoders.PolarDecoder(code_length, info_bits, channel)

    data = encoder.encode_messages(data)
    data = modulator.modulate_symbols(data)
    data = channel.transmit_data(data)
    # data = demodulator.demodulate_symbols(data)
    data = decoder.decode_messages(data)

    return data


def transmit_image(
    img_name: Union[str, Path], code_length: int, info_bits: int, snr_db: int
) -> Tuple[float, float]:
    inp_img, resolution = load_image_raw(img_name)
    inp_bits = unpack_bytes_to_bits(inp_img, info_bits)
    out_bits = transmit_bits(inp_bits, code_length, info_bits, snr_db)
    out_img = pack_bits_to_bytes(out_bits)

    _out_file, _out_ext = os.path.splitext(img_name)
    out_file = "{}_out_{}db{}".format(_out_file, snr_db, _out_ext)
    out_file = Path(out_file).name
    save_image_raw(out_file, out_img)

    ber = float(metrics.ber(inp_bits, out_bits))
    psnr = get_psnr(inp_img, out_img)

    return ber, psnr


if __name__ == "__main__":
    code_length = 256
    info_bits = 8
    snr_dbs = [0, 2, 4, 5, 6, 7, 8, 10, 15]  # dB
    bers = []
    psnrs = []

    image_file = "kodim23.jpeg"
    for snr_db in snr_dbs:
        ber, psnr = transmit_image(image_file, code_length, info_bits, snr_db)
        bers.append(ber)
        psnrs.append(psnr)
        print(f"SNR {snr_db:2d}dB: BER {ber:10.8f}, PSNR {psnr:7.3f}dB")

    plt.plot(snr_dbs, bers)
    plt.yscale("log")
    plt.title("BER")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (-)")
    plt.savefig("ber.png", dpi=300)

    plt.clf()

    plt.plot(snr_dbs, psnrs)
    plt.title("PSNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.savefig("psnr.png", dpi=300)
