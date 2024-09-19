import os.path
import argparse

from imageio import imread, imsave
import numpy as np

from digcommpy.messages import unpack_to_bits, pack_to_dec
from digcommpy.encoders import PolarEncoder
from digcommpy.decoders import PolarDecoder
from digcommpy.channels import BscChannel



def load_image(image_file, num_bits=8, binary=True, **kwargs):
    pixels = imread(image_file)
    resolution = np.shape(pixels)
    pixels = np.reshape(pixels, (-1, ))
    if binary:
        pixels = unpack_to_bits(pixels, num_bits=num_bits)
    return pixels, resolution

def save_image(filepath, image, resolution, binary=True):
    if binary:
        #image = np.reshape(image.ravel(), (-1, 8))
        image = pack_to_dec(image)
    imsave(filepath, image.reshape(resolution))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file")
    parser.add_argument("-b", "--bits", dest="num_bits", type=int, default=8)
    parser.add_argument("-p", type=float, default=.2)
    parser.add_argument("-n", type=int, default=None)
    args = vars(parser.parse_args())
    _out_file, _out_ext = os.path.splitext(args['image_file'])
    out_file = "{}-bsc{}{}".format(_out_file, args['p'], _out_ext)
    data, resolution = load_image(**args)
    channel = BscChannel(args['p'])
    if args['n'] is not None:
        enc = PolarEncoder(args['n'], args['num_bits'], channel)
        data = enc.encode_messages(data)
    rec = channel.transmit_data(data)
    if args['n'] is not None:
        dec = PolarDecoder(args['n'], args['num_bits'], channel)
    save_image(out_file, rec, resolution)

if __name__ == "__main__":
    main()
