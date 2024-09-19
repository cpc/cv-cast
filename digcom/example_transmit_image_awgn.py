import os.path
import argparse
from pathlib import Path

from imageio.v2 import imread, imsave
import numpy as np

from digcommpy.messages import unpack_to_bits, pack_to_dec
from digcommpy.encoders import PolarEncoder, RepetitionEncoder
from digcommpy.decoders import PolarDecoder, RepetitionDecoder
from digcommpy.channels import AwgnChannel, BawgnChannel, BscChannel
from digcommpy.modulators import BpskModulator, QamModulator
from digcommpy.demodulators import BpskDemodulator, QamDemodulator
from digcommpy import metrics


def load_image(image_file, num_bits=8, binary=True, raw=True, **kwargs):
    if raw:
        tmp_pixels = imread(image_file)
        resolution = np.shape(tmp_pixels)
        with open(image_file, "rb") as rf:
            pixels = np.fromfile(rf, dtype=np.uint8)
    else:
        pixels = imread(image_file)
        resolution = np.shape(pixels)
        pixels = np.reshape(pixels, (-1,))

    if binary:
        pixels = unpack_to_bits(pixels, num_bits=num_bits)

    return pixels, resolution


def save_image(filepath, image, resolution, binary=True, raw=True):
    if binary:
        # image = np.reshape(image.ravel(), (-1, 8))
        image = pack_to_dec(image)

    if raw:
        with open(filepath, "wb") as wf:
            image.astype("uint8").tofile(wf)
    else:
        imsave(filepath, image.reshape(resolution).astype("uint8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file")
    parser.add_argument(
        "-b",
        "--bits",
        dest="num_bits",
        type=int,
        default=8,
        help="Number of information bits",
    )
    parser.add_argument("-s", "--snr-db", type=int, default=10)
    parser.add_argument("-n", type=int, default=256, help="Code word length")
    args = vars(parser.parse_args())

    _out_file, _out_ext = os.path.splitext(args["image_file"])
    out_file = "{}-awgn{}".format(_out_file, _out_ext)
    out_file = Path(out_file).name

    raw = True
    data, resolution = load_image(**args, raw=raw)
    inp_data = data

    m = 4
    modulator = BpskModulator()
    demodulator = BpskDemodulator()
    # channel = BscChannel(0.3)
    # channel = BawgnChannel(snr_db=args["snr_db"], rate=args["num_bits"] / args["n"])
    channel = BawgnChannel(snr_db=args["snr_db"])

    if args["n"] is not None:
        enc = PolarEncoder(16, 4, "BAWGN", 0.0)
        data = enc.encode_messages(data)

    # data = modulator.modulate_symbols(data)#, m=m)
    data = channel.transmit_data(data)
    # data = demodulator.demodulate_symbols(data)#, m=m)

    if args["n"] is not None:
        dec = PolarDecoder(16, 4, "BAWGN", 0.0)
        out_data = dec.decode_messages(data)

    save_image(out_file, out_data, resolution, raw=raw)
    # print(inp_data)
    # print(data)
    # out_data, _ = load_image(image_file=out_file, raw=raw)
    # print(type(out_data))
    ber = metrics.ber(inp_data, out_data)
    print("The BER is {}".format(ber))


if __name__ == "__main__":
    main()
