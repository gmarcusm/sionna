#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from abc import abstractmethod
from typing import Optional, Tuple
import torch
from sionna.phy.object import Object


class ChannelModel(Object):
    # pylint: disable=line-too-long
    r"""
    Abstract class that defines an interface for channel models.

    Any channel model which generates channel impulse responses
    must implement this interface.
    All the channel models available in Sionna,
    such as :class:`~sionna.phy.channel.RayleighBlockFading`
    or :class:`~sionna.phy.channel.tr38901.TDL`, implement this interface.

    *Remark:* Some channel models only require a subset of the input parameters.

    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`.
        Batch size.

    :input num_time_steps: `int`.
        Number of time steps.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :output a: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx, num_tx, num_paths], `torch.float`.
        Path delays [s].

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel import RayleighBlockFading

        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=1,
            num_tx=1, num_tx_ant=1
        )
        a, tau = channel_model(batch_size=64, num_time_steps=1,
                               sampling_frequency=1e6)
        print(a.shape)
        # torch.Size([64, 1, 1, 1, 1, 1, 1])
        print(tau.shape)
        # torch.Size([64, 1, 1, 1])
    """

    def __init__(
        self, precision: Optional[str] = None, device: Optional[str] = None, **kwargs
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

    @abstractmethod
    def __call__(
        self, batch_size: int, num_time_steps: int, sampling_frequency: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
