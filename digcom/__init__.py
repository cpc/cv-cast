import math

import matplotlib.pyplot as plt
import numpy as np
import sionna as sn
import tensorflow as tf
from tensorflow import keras


def noop(signal):
    return signal


class OFDMSystemRaw(keras.Model):  # Inherits from Keras Model
    # Values match those of PUSCH transmitter, except pilot pattern
    # https://nvlabs.github.io/sionna/examples/5G_NR_PUSCH.html
    RESOURCE_GRID_CONFIG: dict = {
        "num_ofdm_symbols": 14,
        "fft_size": 192,
        "subcarrier_spacing": 30e3,
        "cyclic_prefix_length": 17,
        "pilot_pattern": "kronecker",
        "pilot_ofdm_symbol_indices": [2, 11],
    }

    def __init__(
        self,
        batch_size,
        resource_grid_config_override: dict = {},
        num_ut: int = 1,
        num_ut_ant: int = 1,
        num_bs: int = 1,
        num_bs_ant: int = 4,
        digital: bool = False,
        perfect_csi: bool = False,
        coderate: float = 1.0,
        no_coding: bool = False,
        nbits_per_sym: int = 2,
        show: bool = False,
    ):
        super().__init__()

        self.NUM_UT = num_ut
        self.NUM_BS = num_bs
        self.NUM_UT_ANT = num_ut_ant
        self.NUM_BS_ANT = num_bs_ant
        self.NUM_STREAMS_PER_TX = self.NUM_UT_ANT
        self.CARRIER_FREQUENCY = 2.6e9  # Carrier frequency in Hz.
        self.digital = digital
        self.perfect_csi = perfect_csi
        self.BATCH_SIZE = batch_size
        self.coderate = coderate
        self.show = show

        self.RESOURCE_GRID_CONFIG.update(resource_grid_config_override)

        self._set_antennas(show=show)
        self._set_resource_grid(self.RESOURCE_GRID_CONFIG, show=show)

        self.num_mapped_symbols = 0  # symbols after constellation mapping before mapping to resource grid
        self.num_transmitted_symbols = 0  # symbols after resource grid mapping entering the channel

        # needed also for noise variance computation
        self.NUM_BITS_PER_SYMBOL = nbits_per_sym  # 2 = 4QAM

        if self.digital:
            n = int(
                self.RESOURCE_GRID.num_data_symbols * self.NUM_BITS_PER_SYMBOL
            )  # Number of coded bits
            k = int(n * self.coderate)  # Number of information bits

            if k != (n * self.coderate):
                raise ValueError(f"Invalid k: {n * self.coderate}")

            self.k = k

            self.binary_source = sn.utils.BinarySource()
            if no_coding:
                self.encoder = noop
            else:
                self.encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
            self.mapper = sn.mapping.Mapper("qam", self.NUM_BITS_PER_SYMBOL)

        self._set_channel_model(show=show)
        self._set_channel()

        if self.digital:
            self.demapper = sn.mapping.Demapper("app", "qam", self.NUM_BITS_PER_SYMBOL)
            if no_coding:
                self.decoder = noop
            else:
                self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)

    # @tf.function  # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        """Transmit random signal over the channel"""

        if self.digital:
            bits = self.binary_source(
                [batch_size, self.NUM_UT, self.RESOURCE_GRID.num_streams_per_tx, self.k]
            )
            codewords = self.encoder(bits)
            x = self.mapper(codewords)
        else:
            shape = (
                self.BATCH_SIZE,
                self.NUM_UT,
                self.NUM_UT_ANT,
                self.RESOURCE_GRID.num_data_symbols,
            )
            re = tf.random.normal(shape)
            im = tf.random.normal(shape)
            x = tf.complex(re, im)

        out = self.run_with_input(x, ebno_db)

        if self.digital:
            return bits, out
        else:
            return x, out

    # @tf.function
    def run_with_input(self, x, ebno_db):
        """Transmit x over the channel"""

        no = sn.utils.ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
            coderate=self.coderate,
            resource_grid=self.RESOURCE_GRID,
        )

        # Transmitter
        if self.digital:
            x_cw = self.encoder(x)
            x_map = self.mapper(x_cw)
        else:
            x_map = x

        self.num_mapped_symbols = x_map.shape

        x_rg = self.rg_mapper(x_map)

        self.num_transmitted_symbols = x_rg.shape

        # Channel
        y, h_freq = self.channel([x_rg, no])

        # Receiver
        if self.perfect_csi:
            h_hat, err_var = h_freq, 0.0
        else:
            h_hat, err_var = self.ls_est([y, no])

        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])

        if self.digital:
            llr = self.demapper([x_hat, no_eff])
            bits_hat = self.decoder(llr)
            return bits_hat
        else:
            return x_hat

    def _set_antennas(self, show: bool = False):
        self.RX_TX_ASSOCIATION = np.array([[1]])
        self.STREAM_MANAGEMENT = sn.mimo.StreamManagement(
            self.RX_TX_ASSOCIATION, self.NUM_STREAMS_PER_TX
        )

        self.UT_ARRAY = sn.channel.tr38901.Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=self.CARRIER_FREQUENCY,
        )

        self.BS_ARRAY = sn.channel.tr38901.AntennaArray(
            num_rows=1,
            num_cols=int(self.NUM_BS_ANT / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",  # Try 'omni'
            carrier_frequency=self.CARRIER_FREQUENCY,
        )

        if show:
            self.UT_ARRAY.show()
            plt.title("UT Array")
            self.BS_ARRAY.show()
            plt.title("BS Array")
            plt.show()

    def _set_resource_grid(self, config: dict, show: bool = False):
        self.RESOURCE_GRID = sn.ofdm.ResourceGrid(
            **config,
            num_tx=self.NUM_UT,
            num_streams_per_tx=self.NUM_STREAMS_PER_TX,
            # pilot_pattern="kronecker",
            # num_ofdm_symbols=14,
            # fft_size=76,
            # subcarrier_spacing=30e3,
            # cyclic_prefix_length=6,
            # pilot_ofdm_symbol_indices=[2, 11],
        )

        if show:
            self.RESOURCE_GRID.show()
            self.RESOURCE_GRID.pilot_pattern.show()
            plt.show()

        self.rg_mapper = sn.ofdm.ResourceGridMapper(self.RESOURCE_GRID)

    def _set_channel_model(self, show: bool = False):
        self.DIRECTION = "uplink"  # In the `uplink`, the UT is transmitting.

        self.channel_model = sn.channel.tr38901.UMa(
            carrier_frequency=self.CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=self.UT_ARRAY,
            bs_array=self.BS_ARRAY,
            direction=self.DIRECTION,
        )

        self.topology = sn.channel.gen_single_sector_topology(
            batch_size=self.BATCH_SIZE, num_ut=self.NUM_UT, scenario="uma"
        )
        (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
        ) = self.topology

        self.channel_model.set_topology(
            ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state
        )

        if show:
            self.channel_model.show_topology()
            plt.show()

    def _set_channel(self):
        self.channel = sn.channel.OFDMChannel(
            self.channel_model,
            self.RESOURCE_GRID,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True,
        )

        self.ls_est = sn.ofdm.LSChannelEstimator(
            self.RESOURCE_GRID, interpolation_type="nn"
        )

        self.lmmse_equ = sn.ofdm.LMMSEEqualizer(
            self.RESOURCE_GRID, self.STREAM_MANAGEMENT
        )
