#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes for the simulation of flat-fading channels"""

from typing import Optional, Tuple, Union

import torch

from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.channel.spatial_correlation import SpatialCorrelation
from sionna.phy.config import Precision
from sionna.phy.utils import complex_normal

__all__ = [
    "GenerateFlatFadingChannel",
    "ApplyFlatFadingChannel",
    "FlatFadingChannel",
]


class GenerateFlatFadingChannel(Block):
    r"""Generates tensors of flat-fading channel realizations

    This class generates batches of random flat-fading channel matrices.
    A spatial correlation can be applied.

    :param num_tx_ant: Number of transmit antennas
    :param num_rx_ant: Number of receive antennas
    :param spatial_corr: Spatial correlation to be applied.
        Defaults to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`.
        Number of channel matrices to generate.

    :output h: [batch_size, num_rx_ant, num_tx_ant], `torch.complex`.
        Batch of random flat fading channel matrices.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import GenerateFlatFadingChannel

        gen_chn = GenerateFlatFadingChannel(num_tx_ant=4, num_rx_ant=16)
        h = gen_chn(batch_size=32)
        print(h.shape)
        # torch.Size([32, 16, 4])
    """

    def __init__(
        self,
        num_tx_ant: int,
        num_rx_ant: int,
        spatial_corr: Optional[SpatialCorrelation] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self.spatial_corr = spatial_corr

    @property
    def spatial_corr(self) -> Optional[SpatialCorrelation]:
        r"""Get/set spatial correlation to be applied"""
        return self._spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value: Optional[SpatialCorrelation]) -> None:
        self._spatial_corr = value

    def call(self, batch_size: int) -> torch.Tensor:
        """Generate batch of random flat fading channel matrices.

        :param batch_size: Number of channel matrices to generate

        :output h: Batch of random flat fading channel matrices
        """
        # Generate standard complex Gaussian matrices with unit variance
        # Uses smart random that switches to global RNG in compiled mode for graph fusion
        shape = [batch_size, self._num_rx_ant, self._num_tx_ant]
        h = complex_normal(
            shape,
            precision=self.precision,
            device=self.device,
            generator=self.torch_rng,
        )

        # Apply spatial correlation
        if self.spatial_corr is not None:
            h = self.spatial_corr(h)

        return h


class ApplyFlatFadingChannel(Block):
    r"""Applies given channel matrices to a vector input and adds AWGN

    This class applies a given tensor of flat-fading channel matrices
    to an input tensor. AWGN noise can be optionally added.
    Mathematically, for channel matrices
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}`
    and input :math:`\mathbf{x}\in\mathbb{C}^{K}`, the output is

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})`
    is an AWGN vector that is optionally added.

    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [batch_size, num_tx_ant], `torch.complex`.
        Transmit vectors.

    :input h: [batch_size, num_rx_ant, num_tx_ant], `torch.complex`.
        Channel realizations. Will be broadcast to the
        dimensions of ``x`` if needed.

    :input no: `None` (default) | `torch.Tensor`, `torch.float`.
        (Optional) noise power ``no`` per complex dimension.
        Will be broadcast to the shape of ``y``.
        For more details, see :class:`~sionna.phy.channel.AWGN`.

    :output y: [batch_size, num_rx_ant], `torch.complex`.
        Channel output.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import ApplyFlatFadingChannel

        app_chn = ApplyFlatFadingChannel()
        x = torch.randn(32, 4, dtype=torch.complex64)
        h = torch.randn(32, 16, 4, dtype=torch.complex64)
        y = app_chn(x, h)
        print(y.shape)
        # torch.Size([32, 16])
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
        h: torch.Tensor,
        no: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply flat fading channel to input.

        :param x: Transmit vectors
        :param h: Channel realizations
        :param no: Optional noise power per complex dimension

        :output y: Channel output
        """
        y = (h @ x.unsqueeze(-1)).squeeze(-1)

        if no is not None:
            y = self._awgn(y, no)

        return y


class FlatFadingChannel(Block):
    r"""Applies random channel matrices to a vector input and adds AWGN

    This class combines :class:`~sionna.phy.channel.GenerateFlatFadingChannel` and
    :class:`~sionna.phy.channel.ApplyFlatFadingChannel` and computes the output of
    a flat-fading channel with AWGN.

    For a given batch of input vectors :math:`\mathbf{x}\in\mathbb{C}^{K}`,
    the output is

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` are randomly generated
    flat-fading channel matrices and
    :math:`\mathbf{n}\in\mathbb{C}^{M}\sim\mathcal{CN}(0, N_o\mathbf{I})`
    is an AWGN vector that is optionally added.

    A :class:`~sionna.phy.channel.SpatialCorrelation` can be configured and the
    channel realizations optionally returned. This is useful to simulate
    receiver algorithms with perfect channel knowledge.

    :param num_tx_ant: Number of transmit antennas
    :param num_rx_ant: Number of receive antennas
    :param spatial_corr: Spatial correlation to be applied.
        Defaults to `None`.
    :param return_channel: Indicates if the channel realizations should be
        returned. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [batch_size, num_tx_ant], `torch.complex`.
        Tensor of transmit vectors.

    :input no: `None` (default) | `torch.Tensor`, `torch.float`.
        (Optional) noise power ``no`` per complex dimension.
        Will be broadcast to the shape of ``y``.
        For more details, see :class:`~sionna.phy.channel.AWGN`.

    :output y: [batch_size, num_rx_ant], `torch.complex`.
        Channel output.

    :output h: [batch_size, num_rx_ant, num_tx_ant], `torch.complex`.
        Channel realizations. Will only be returned if
        ``return_channel==True``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import FlatFadingChannel

        chn = FlatFadingChannel(num_tx_ant=4, num_rx_ant=16, return_channel=True)
        x = torch.randn(32, 4, dtype=torch.complex64)
        y, h = chn(x)
        print(y.shape)
        # torch.Size([32, 16])
        print(h.shape)
        # torch.Size([32, 16, 4])
    """

    def __init__(
        self,
        num_tx_ant: int,
        num_rx_ant: int,
        spatial_corr: Optional[SpatialCorrelation] = None,
        return_channel: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._return_channel = return_channel
        self._gen_chn = GenerateFlatFadingChannel(
            self._num_tx_ant,
            self._num_rx_ant,
            spatial_corr,
            precision=precision,
            device=device,
        )
        self._app_chn = ApplyFlatFadingChannel(precision=precision, device=device)

    @property
    def spatial_corr(self) -> Optional[SpatialCorrelation]:
        r"""Get/set spatial correlation to be applied"""
        return self._gen_chn.spatial_corr

    @spatial_corr.setter
    def spatial_corr(self, value: Optional[SpatialCorrelation]) -> None:
        self._gen_chn.spatial_corr = value

    @property
    def generate(self) -> GenerateFlatFadingChannel:
        r"""Access the internal :class:`GenerateFlatFadingChannel`"""
        return self._gen_chn

    @property
    def apply(self) -> ApplyFlatFadingChannel:
        r"""Access the internal :class:`ApplyFlatFadingChannel`"""
        return self._app_chn

    def call(
        self,
        x: torch.Tensor,
        no: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply flat fading channel to input.

        :param x: Transmit vectors
        :param no: Optional noise power per complex dimension

        :output y: Channel output.

        :output h: (Optional) Channel realizations. Returned only if
            ``return_channel`` is `True`.
        """
        # Generate a batch of channel realizations
        batch_size = x.shape[0]
        h = self._gen_chn(batch_size)

        # Apply the channel to the input
        y = self._app_chn(x, h, no)

        if self._return_channel:
            return y, h
        else:
            return y
