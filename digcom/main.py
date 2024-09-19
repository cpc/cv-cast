import math

import numpy as np
import matplotlib.pyplot as plt

from crc import Calculator, Crc8

import commpy
# https://github.com/veeresht/CommPy/blob/master/commpy/wifi80211.py
from commpy.wifi80211 import Wifi80211

from digcommpy import messages, encoders, decoders, channels, modulators, metrics

def commpy_example():
    # https://github.com/veeresht/CommPy/blob/master/commpy/examples/wifi80211_conv_encode_decode.py
    channels = commpy.channels.SISOFlatChannel(None, (1 + 0j, 0j))

    # MCS 0 : BPSK 1/2
    # MCS 1 : QPSK 1/2
    # MCS 2 : QPSK 3/4
    # MCS 3 : 16-QAM 1/2
    # MCS 4 : 16-QAM 3/4
    # MCS 5 : 64-QAM 2/3
    # MCS 6 : 64-QAM 3/4
    # MCS 7 : 64-QAM 5/6
    # MCS 8 : 256-QAM 3/4
    # MCS 9 : 256-QAM 5/6
    w2 = Wifi80211(mcs=2)
    w3 = Wifi80211(mcs=5)

    # SNR range to test
    SNRs2 = np.arange(0, 6) + 10 * math.log10(w2.get_modem().num_bits_symbol)
    SNRs3 = np.arange(0, 6) + 10 * math.log10(w3.get_modem().num_bits_symbol)

    # returns tuple of 4:
    # BERs (1D): Estimated Bit Error Ratio corresponding to each SNRs
    # BEs  (2D): Number of Estimated Bits with Error per transmission corresponding to each SNRs
    # CEs  (2D): Number of Estimated Chunks with Errors per transmission corresponding to each SNRs
    # NCs  (2D): Number of Chunks transmitted per transmission corresponding to each SNRs
    BERs_mcs2 = w2.link_performance(
        channels, SNRs2, 10, 10, 600, stop_on_surpass_error=False
    )
    BERs_mcs3 = w3.link_performance(
        channels, SNRs3, 10, 10, 600, stop_on_surpass_error=False
    )

    print(SNRs2.shape)
    print(len(BERs_mcs2))

    # Test
    plt.semilogy(SNRs2, BERs_mcs2[0], "o-")
    plt.semilogy(SNRs3, BERs_mcs3[0], "o-")
    plt.grid()
    plt.xlabel("Signal to Noise Ration (dB)")
    plt.ylabel("Bit Error Rate")
    plt.legend(("MCS 2", "MCS 3"))
    plt.show()

def digcommpy_exmaple():
    # Parameters
    n, k = 16, 4
    snr = 20.  # dB
    # Blocks
    encoder = encoders.PolarEncoder(n, k, "BAWGN", snr)
    modulator = modulators.BpskModulator()
    # modulator = modulators.QamModulator()
    channel = channels.BawgnChannel(snr, rate=k/n)
    decoder = decoders.PolarDecoder(n, k, "BAWGN", snr)
    # Simulation
    mess = messages.generate_data(k, number=1000, binary=True)
    # print(mess)
    codewords = encoder.encode_messages(mess)
    channel_input = modulator.modulate_symbols(codewords)
    print(channel_input)
    channel_output = channel.transmit_data(channel_input)
    est_mess = decoder.decode_messages(channel_output)
    ber = metrics.ber(mess, est_mess)
    print("The BER is {}".format(ber))

if __name__ == "__main__":
    data = bytes([0, 1, 2, 3, 4, 5, 6, 7])
    crc_calculator = Calculator(Crc8.CCITT, optimized=True)

    checksum = crc_calculator.checksum(data)

    print("checksum: ", checksum)

    digcommpy_exmaple()
