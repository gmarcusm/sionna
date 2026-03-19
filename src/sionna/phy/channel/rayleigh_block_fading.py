#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class for simulating Rayleigh block fading"""

from typing import Optional, Tuple

import torch

from sionna.phy.utils import normal
from .channel_model import ChannelModel

__all__ = ["RayleighBlockFading"]


class RayleighBlockFading(ChannelModel):
    r"""Generates channel impulse responses corresponding to a Rayleigh block
    fading channel model

    The channel impulse responses generated are formed of a single path with
    zero delay and a normally distributed fading coefficient.
    All time steps of a batch example share the same channel coefficient
    (block fading).

    This class can be used in conjunction with the classes that simulate the
    channel response in time or frequency domain, i.e.,
    :class:`~sionna.phy.channel.OFDMChannel`,
    :class:`~sionna.phy.channel.TimeChannel`,
    :class:`~sionna.phy.channel.GenerateOFDMChannel`,
    :class:`~sionna.phy.channel.ApplyOFDMChannel`,
    :class:`~sionna.phy.channel.GenerateTimeChannel`,
    :class:`~sionna.phy.channel.ApplyTimeChannel`.

    :param num_rx: Number of receivers (:math:`N_R`)
    :param num_rx_ant: Number of antennas per receiver (:math:`N_{RA}`)
    :param num_tx: Number of transmitters (:math:`N_T`)
    :param num_tx_ant: Number of antennas per transmitter (:math:`N_{TA}`)
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is
        used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`.
        Batch size.

    :input num_time_steps: `int`.
        Number of time steps.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz]. Not used but accepted for compatibility with
        the :class:`~sionna.phy.channel.ChannelModel` interface.

    :output a: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths=1, num_time_steps], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx, num_tx, num_paths=1], `torch.float`.
        Path delays [s].

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import RayleighBlockFading

        channel_model = RayleighBlockFading(num_rx=1, num_rx_ant=2,
                                            num_tx=1, num_tx_ant=4)
        h, tau = channel_model(batch_size=32, num_time_steps=14)
        print(h.shape)
        # torch.Size([32, 1, 2, 1, 4, 1, 14])
        print(tau.shape)
        # torch.Size([32, 1, 1, 1])
    """

    def __init__(
        self,
        num_rx: int,
        num_rx_ant: int,
        num_tx: int,
        num_tx_ant: int,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._num_tx = num_tx
        self._num_tx_ant = num_tx_ant
        self._num_rx = num_rx
        self._num_rx_ant = num_rx_ant

    @property
    def num_tx(self) -> int:
        """Number of transmitters"""
        return self._num_tx

    @property
    def num_tx_ant(self) -> int:
        """Number of antennas per transmitter"""
        return self._num_tx_ant

    @property
    def num_rx(self) -> int:
        """Number of receivers"""
        return self._num_rx

    @property
    def num_rx_ant(self) -> int:
        """Number of antennas per receiver"""
        return self._num_rx_ant

    def __call__(
        self,
        batch_size: int,
        num_time_steps: int,
        sampling_frequency: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate channel impulse response for Rayleigh block fading."""
        # Delays: single path with zero delay
        delays = torch.zeros(
            batch_size,
            self._num_rx,
            self._num_tx,
            1,  # Single path
            dtype=self.dtype,
            device=self.device,
        )

        # Fading coefficients: complex Gaussian with unit variance
        std = torch.tensor(0.5, dtype=self.dtype, device=self.device).sqrt()
        shape = [
            batch_size,
            self._num_rx,
            self._num_rx_ant,
            self._num_tx,
            self._num_tx_ant,
            1,  # One path
            1,  # Same response over the block
        ]

        # Uses smart normal that switches to global RNG in compiled mode
        h_real = normal(
            mean=0.0,
            std=std,
            size=shape,
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        h_imag = normal(
            mean=0.0,
            std=std,
            size=shape,
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        h = torch.complex(h_real, h_imag)

        # Tile the response over all time steps (block fading)
        h = h.expand(-1, -1, -1, -1, -1, -1, num_time_steps)

        return h, delays
