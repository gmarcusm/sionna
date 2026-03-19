#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block for applying OFDM channel: single-tap channel response in the frequency domain"""

from typing import Optional, Union

import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.utils import expand_to_rank
from .awgn import AWGN

__all__ = ["ApplyOFDMChannel"]


class ApplyOFDMChannel(Block):
    r"""Apply single-tap channel frequency responses to channel inputs

    For each OFDM symbol :math:`s` and subcarrier :math:`n`, the single-tap channel
    is applied as follows:

    .. math::
        y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}

    where :math:`y_{s,n}` is the channel output computed by this layer,
    :math:`\widehat{h}_{s, n}` the frequency channel response (``h_freq``),
    :math:`x_{s,n}` the channel input ``x``, and :math:`w_{s,n}` the additive noise.

    For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
    of each receiver and by summing over all the antennas of all transmitters.

    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel inputs.

    :input h_freq: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel frequency responses.

    :input no: `None` (default) | Tensor, `torch.float`.
        Tensor whose shape can be broadcast to the shape of the
        channel outputs:
        [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size].
        The (optional) noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the outputs.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of the channel outputs. This allows, e.g., adding noise of
        different variance to each example in a batch. If ``no`` has a lower
        rank than the channel outputs, then ``no`` will be broadcast to the
        shape of the channel outputs by adding dummy dimensions after the
        last axis.

    :output y: [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel outputs.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import ApplyOFDMChannel

        apply_ch = ApplyOFDMChannel()

        # Create dummy inputs
        batch_size, num_tx, num_tx_ant = 16, 2, 4
        num_rx, num_rx_ant = 1, 8
        num_ofdm_symbols, fft_size = 14, 64

        x = torch.randn(batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size,
                        dtype=torch.complex64)
        h_freq = torch.randn(batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
                             num_ofdm_symbols, fft_size, dtype=torch.complex64)

        y = apply_ch(x, h_freq)
        print(y.shape)
        # torch.Size([16, 1, 8, 14, 64])
    """

    def __init__(
        self,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._awgn = AWGN(precision=self.precision, device=self.device)

    def call(
        self,
        x: torch.Tensor,
        h_freq: torch.Tensor,
        no: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply OFDM channel frequency response to input.

        :param x: Channel inputs
        :param h_freq: Channel frequency responses
        :param no: Optional noise power per complex dimension

        :output y: Channel outputs
        """
        # Apply the channel response
        # x: [batch, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        # h_freq: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        # Expand x to match h_freq rank by adding num_rx and num_rx_ant dimensions
        x = expand_to_rank(x, h_freq.dim(), axis=1)
        # x is now: [batch, 1, 1, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

        # Element-wise multiply and sum over num_tx_ant (dim=4) and num_tx (dim=3)
        y = (h_freq * x).sum(dim=4).sum(dim=3)
        # y: [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # Add AWGN if requested
        if no is not None:
            y = self._awgn(y, no)

        return y
