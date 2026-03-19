#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block for simulating an AWGN channel"""

from typing import Optional, Union
import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from sionna.phy.utils import expand_to_rank, complex_normal

__all__ = ["AWGN"]


class AWGN(Block):
    r"""Add complex AWGN to the inputs with a certain variance.

    This block adds complex AWGN noise with variance ``no`` to the input.
    The noise has variance ``no/2`` per real dimension.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [...], `torch.complex`.
        Channel input.

    :input no: Scalar or Tensor, `torch.float`.
        Scalar or tensor whose shape can be broadcast to the shape of ``x``.
        The noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the input.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of ``x``. This allows, e.g., adding noise of different
        variance to each example in a batch. If ``no`` has a lower rank than
        ``x``, then ``no`` will be broadcast to the shape of ``x`` by adding
        dummy dimensions after the last axis.

    :output y: Tensor with same shape as ``x``, `torch.complex`.
        Channel output.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import AWGN

        awgn_channel = AWGN()
        x = torch.randn(64, 16, dtype=torch.complex64)
        no = 0.1
        y = awgn_channel(x, no)
        print(y.shape)
        # torch.Size([64, 16])
    """

    def __init__(
        self,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

    def call(
        self,
        x: torch.Tensor,
        no: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Apply AWGN to the input."""
        # Create tensor of complex-valued Gaussian noise with unit variance
        # Uses smart random that switches to global RNG in compiled mode for graph fusion
        noise = complex_normal(x.shape, precision=self.precision, device=self.device,
                               generator=self.torch_rng)

        # Convert no to tensor if it's a scalar
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, dtype=self.dtype, device=self.device)

        # Add extra dimensions for broadcasting
        no = expand_to_rank(no, x.dim(), axis=-1)

        # Apply variance scaling
        no = no.to(dtype=self.dtype, device=self.device)
        noise = noise * no.sqrt().to(dtype=self.cdtype)

        # Add noise to input
        return x + noise
