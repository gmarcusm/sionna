#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH Receiver for the 5G NR module of Sionna PHY"""

from typing import Optional, Tuple, Union
import numpy as np
import torch

import sionna
from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import OFDMDemodulator, LinearDetector
from sionna.phy.utils import insert_dims
from sionna.phy.channel import time_to_ofdm_channel


__all__ = ["PUSCHReceiver"]


class PUSCHReceiver(Block):
    r"""This block implements a full receiver for batches of 5G NR PUSCH slots
    sent by multiple transmitters. Inputs can be in the time or frequency
    domain. Perfect channel state information can be optionally provided.
    Different channel estimators, MIMO detectors, and transport decoders
    can be configured.

    The block combines multiple processing blocks into a single block.
    Blocks with dashed lines are optional and depend on the configuration.

    If the ``input_domain`` equals "time", the inputs :math:`\mathbf{y}` are
    first transformed to resource grids with the
    :class:`~sionna.phy.ofdm.OFDMDemodulator`. Then channel estimation is
    performed, e.g., with the help of the
    :class:`~sionna.phy.nr.PUSCHLSChannelEstimator`. If
    ``channel_estimator`` is chosen to be "perfect", this step is skipped and
    the input :math:`\mathbf{h}` is used instead. Next, MIMO detection is
    carried out with an arbitrary
    :class:`~sionna.phy.ofdm.OFDMDetector`. The resulting LLRs for each layer
    are then combined to transport blocks with the help of the
    :class:`~sionna.phy.nr.LayerDemapper`. Finally, the transport blocks are
    decoded with the :class:`~sionna.phy.nr.TBDecoder`.

    :param pusch_transmitter: Transmitter used for the generation of the
        transmit signals.
    :param channel_estimator: Channel estimator to be used.
        If `None`, the :class:`~sionna.phy.nr.PUSCHLSChannelEstimator` with
        linear interpolation is used.
        If "perfect", no channel estimation is performed and the channel
        state information ``h`` must be provided as additional input.
        Defaults to `None`.
    :param mimo_detector: MIMO detector to be used.
        If `None`, the :class:`~sionna.phy.ofdm.LinearDetector` with
        LMMSE detection is used.
        Defaults to `None`.
    :param tb_decoder: Transport block decoder to be used.
        If `None`, the :class:`~sionna.phy.nr.TBDecoder` with its
        default settings is used.
        Defaults to `None`.
    :param return_tb_crc_status: If `True`, the status of the transport block
        CRC is returned as additional output. Defaults to `False`.
    :param stream_management: Stream management configuration to be used.
        If `None`, it is assumed that there is a single receiver
        which decodes all streams of all transmitters.
        Defaults to `None`.
    :param input_domain: Domain of the input signal.
        Defaults to "freq".
    :param l_min: Smallest time-lag for the discrete complex baseband channel.
        Only needed if ``input_domain`` equals "time".
        Defaults to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`, the default device is
        used.

    :input y: [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex` or [batch size, num_rx, num_rx_ant, num_time_samples + l_max - l_min], `torch.complex`.
        Frequency- or time-domain input signal.

    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :input h: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], `torch.complex` or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], `torch.complex`.
        Perfect channel state information in either frequency or time domain
        (depending on ``input_domain``) to be used for detection.
        Only required if ``channel_estimator`` equals "perfect".

    :output b_hat: [batch_size, num_tx, tb_size], `torch.float`.
        Decoded information bits.

    :output tb_crc_status: [batch_size, num_tx], `torch.bool`.
        Transport block CRC status.

    .. rubric:: Examples

    >>> pusch_config = PUSCHConfig()
    >>> pusch_transmitter = PUSCHTransmitter(pusch_config)
    >>> pusch_receiver = PUSCHReceiver(pusch_transmitter)
    >>> channel = AWGN()
    >>> x, b = pusch_transmitter(16)
    >>> no = 0.1
    >>> y = channel([x, no])
    >>> b_hat = pusch_receiver(y, no)
    """

    def __init__(
        self,
        pusch_transmitter,
        channel_estimator=None,
        mimo_detector=None,
        tb_decoder=None,
        return_tb_crc_status: bool = False,
        stream_management=None,
        input_domain: str = "freq",
        l_min: Optional[int] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if input_domain not in ["time", "freq"]:
            raise ValueError("input_domain must be 'time' or 'freq'")
        self._input_domain = input_domain

        self._return_tb_crc_status = return_tb_crc_status
        self._resource_grid = pusch_transmitter.resource_grid

        # (Optionally) Create OFDMDemodulator
        if self._input_domain == "time":
            if l_min is None:
                raise ValueError("l_min must be provided for input_domain='time'")
            self._l_min = l_min
            self._ofdm_demodulator = OFDMDemodulator(
                fft_size=pusch_transmitter._num_subcarriers,
                l_min=self._l_min,
                cyclic_prefix_length=pusch_transmitter._cyclic_prefix_length,
                precision=self.precision,
                device=self.device,
            )
        else:
            self._ofdm_demodulator = None
            self._l_min = None

        # Use or create default ChannelEstimator
        self._perfect_csi = False
        self._w = None
        if channel_estimator is None:
            self._channel_estimator = sionna.phy.nr.PUSCHLSChannelEstimator(
                self.resource_grid,
                pusch_transmitter._dmrs_length,
                pusch_transmitter._dmrs_additional_position,
                pusch_transmitter._num_cdm_groups_without_data,
                interpolation_type='lin',
                precision=self.precision,
                device=self.device,
            )
        elif channel_estimator == "perfect":
            self._perfect_csi = True
            if pusch_transmitter._precoding == "codebook":
                self._w = pusch_transmitter._precoder._w
                self._w = insert_dims(self._w, 2, 1)
            self._channel_estimator = None
        else:
            self._channel_estimator = channel_estimator

        # Use or create default StreamManagement
        if stream_management is None:
            rx_tx_association = np.ones([1, pusch_transmitter._num_tx], bool)
            self._stream_management = StreamManagement(
                rx_tx_association,
                pusch_transmitter._num_layers,
            )
        else:
            self._stream_management = stream_management

        # Use or create default MIMODetector
        if mimo_detector is None:
            self._mimo_detector = LinearDetector(
                "lmmse", "bit", "maxlog",
                pusch_transmitter.resource_grid,
                self._stream_management,
                "qam",
                pusch_transmitter._num_bits_per_symbol,
                precision=self.precision,
                device=self.device,
            )
        else:
            self._mimo_detector = mimo_detector

        # Create LayerDemapper
        self._layer_demapper = sionna.phy.nr.LayerDemapper(
            pusch_transmitter._layer_mapper,
            num_bits_per_symbol=pusch_transmitter._num_bits_per_symbol,
            precision=self.precision,
            device=self.device,
        )

        # Use or create default TBDecoder
        if tb_decoder is None:
            self._tb_decoder = sionna.phy.nr.TBDecoder(
                pusch_transmitter._tb_encoder,
                precision=self.precision,
                device=self.device,
            )
        else:
            self._tb_decoder = tb_decoder

    #########################################
    # Public methods and properties
    #########################################

    @property
    def resource_grid(self):
        """OFDM resource grid underlying the PUSCH transmissions"""
        return self._resource_grid

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received signal.

        :param y: Received signal in time or frequency domain.
        :param no: AWGN variance.
        :param h: Perfect CSI (required if channel_estimator="perfect").

        :output b_hat: Decoded bits, optionally with CRC status.
        """
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, device=self.device, dtype=self.dtype)

        # (Optional) OFDM Demodulation
        if self._ofdm_demodulator is not None:
            y = self._ofdm_demodulator(y)

        # Channel estimation
        if self._perfect_csi:
            if h is None:
                raise ValueError("h must be provided for perfect CSI")

            # Transform time-domain to frequency-domain channel
            if self._input_domain == "time":
                h = time_to_ofdm_channel(h, self.resource_grid, self._l_min)

            if self._w is not None:
                # Reshape h to put channel matrix dimensions last
                # [batch, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
                h = h.permute(0, 1, 3, 5, 6, 2, 4)

                # Multiply by precoding matrices for effective channels
                # [batch, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_streams]
                h = torch.matmul(h, self._w)

                # Reshape back
                # [batch, num_rx, num_rx_ant, num_tx, num_streams, num_ofdm_symbols, fft_size]
                h = h.permute(0, 1, 5, 2, 6, 3, 4)

            h_hat = h
            err_var = torch.zeros(1, dtype=h_hat.real.dtype, device=h_hat.device)
        else:
            h_hat, err_var = self._channel_estimator(y, no)

        # MIMO Detection
        llr = self._mimo_detector(y, h_hat, err_var, no)

        # Layer demapping
        llr = self._layer_demapper(llr)

        # TB Decoding
        b_hat, tb_crc_status = self._tb_decoder(llr)

        if self._return_tb_crc_status:
            return b_hat, tb_crc_status
        return b_hat

