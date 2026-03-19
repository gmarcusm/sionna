#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layer mapping for the 5G NR module of Sionna PHY."""

from typing import List, Optional, Union
import torch

from sionna.phy import Block
from sionna.phy.utils import flatten_last_dims, split_dim


__all__ = ["LayerMapper", "LayerDemapper"]


class LayerMapper(Block):
    r"""Performs MIMO layer mapping of modulated symbols to layers as defined
    in :cite:p:`3GPPTS38211`.

    The LayerMapper supports PUSCH and PDSCH channels and follows the procedure
    as defined in Sec. 6.3.1.3 and Sec. 7.3.1.3 in :cite:p:`3GPPTS38211`, respectively.

    As specified in Tab. 7.3.1.3.-1 :cite:p:`3GPPTS38211`, the LayerMapper expects two
    input streams for multiplexing if more than 4 layers are active (only
    relevant for PDSCH).

    :param num_layers: Number of MIMO layers. Must be between 1 and 8. If
        ``num_layers`` >= 5, a list of two inputs is expected.
    :param verbose: If `True`, additional parameters are printed.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input inputs: [..., n] or [[..., n1], [..., n2]], `torch.complex`.
        Sequence of symbols to be mapped. If ``num_layers`` >= 5, a list of
        two inputs is expected and n1/n2 must be chosen as defined in
        Tab. 7.3.1.3.-1 :cite:p:`3GPPTS38211`.

    :output x_mapped: [..., num_layers, n/num_layers], `torch.complex`.
        Sequence of symbols mapped to the MIMO layers.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.nr import LayerMapper

        mapper = LayerMapper(num_layers=2)
        symbols = torch.randn(10, 100) + 1j * torch.randn(10, 100)
        mapped = mapper(symbols)
        print(mapped.shape)
        # torch.Size([10, 2, 50])
    """

    def __init__(
        self,
        num_layers: int = 1,
        verbose: bool = False,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool")
        self._verbose = verbose

        if num_layers not in range(1, 9):
            raise ValueError("num_layers must be between 1 and 8.")
        self._num_layers = num_layers

        # Follow Tab. 7.3.1.3-1 from 38.211 for CW multiplexing
        if self._num_layers < 5:
            self._num_codewords = 1
            self._num_layers0 = num_layers
            self._num_layers1 = 0
        elif self._num_layers == 5:
            self._num_codewords = 2
            self._num_layers0 = 2
            self._num_layers1 = 3
        elif self._num_layers == 6:
            self._num_codewords = 2
            self._num_layers0 = 3
            self._num_layers1 = 3
        elif self._num_layers == 7:
            self._num_codewords = 2
            self._num_layers0 = 3
            self._num_layers1 = 4
        elif self._num_layers == 8:
            self._num_codewords = 2
            self._num_layers0 = 4
            self._num_layers1 = 4
        else:
            raise ValueError("Invalid number of layers.")

        if self._verbose:
            print("Number of layers: ", self._num_layers)
            if self._num_codewords == 2:
                print(
                    "Dual codeword mode active and cw multiplexing as "
                    "defined in Tab. 7.3.1.3-1 from 38.211 applied."
                )
                print(f"Length of cw1/cw2: {self._num_layers0}/{self._num_layers1}")

    @property
    def num_codewords(self) -> int:
        """`int` : Number of input codewords for layer mapping.
        Can be either 1 or 2."""
        return self._num_codewords

    @property
    def num_layers(self) -> int:
        """`int` : Number of MIMO layers."""
        return self._num_layers

    @property
    def num_layers0(self) -> int:
        """`int` : Number of layers for first codeword (only relevant for
        `num_codewords` = 2)."""
        if self._num_codewords == 1:
            return self._num_layers
        return self._num_layers0

    @property
    def num_layers1(self) -> int:
        """`int` : Number of layers for second codeword (only relevant for
        `num_codewords` = 2)."""
        if self._num_codewords == 1:
            return 0  # No second stream
        return self._num_layers1

    def build(self, input_shapes):
        """Test input shapes for consistency."""
        if self._num_codewords == 1:  # Single cw mode
            if isinstance(input_shapes, list) and len(input_shapes) == 2:
                if isinstance(input_shapes[0], (list, tuple, torch.Size)):
                    raise ValueError("Only single input codeword expected.")
            if input_shapes[-1] % self._num_layers != 0:
                raise ValueError(
                    "Invalid input dimensions: last dimension must be a "
                    "multiple of num_layers."
                )
        else:  # Dual cw mode
            # Inputs must be a list of two streams
            if not isinstance(input_shapes, list) or len(input_shapes) != 2:
                raise ValueError("List of two input streams is expected.")
            s0 = input_shapes[0]
            s1 = input_shapes[1]

            if s0[-1] % self._num_layers0 != 0:
                raise ValueError(
                    "Invalid input dimensions: last dimension of first input "
                    "must be a multiple of num_layers0."
                )
            if s1[-1] % self._num_layers1 != 0:
                raise ValueError(
                    "Invalid input dimensions: last dimension of second input "
                    "must be a multiple of num_layers1."
                )

            # Verify that length of tb1 and tb2 fit together
            if s0[-1] / self._num_layers0 != s1[-1] / self._num_layers1:
                raise ValueError(
                    f"Invalid input dimensions: length of first input must be "
                    f"{self._num_layers0 / self._num_layers1:.2f} of the length "
                    f"of the second input."
                )

    def call(
        self, inputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Applies MIMO Layer mapping as defined in Sec. 6.3.1.3 and Sec.
        7.3.1.3 38.211.

        :param inputs: Symbol sequence to be mapped.

        :output x_mapped: Symbols mapped to MIMO layers.
        """
        if self._num_codewords == 1:
            s = inputs.shape[-1]
            y = split_dim(
                inputs,
                (int(s / self._num_layers), self._num_layers),
                axis=len(inputs.shape) - 1,
            )
        else:
            # For PDSCH only: support dual stream multiplexing
            x0 = inputs[0]
            x1 = inputs[1]
            s0 = x0.shape[-1]
            s1 = x1.shape[-1]

            y0 = split_dim(
                x0,
                (int(s0 / self._num_layers0), self._num_layers0),
                axis=len(x0.shape) - 1,
            )
            y1 = split_dim(
                x1,
                (int(s1 / self._num_layers1), self._num_layers1),
                axis=len(x1.shape) - 1,
            )

            y = torch.cat([y0, y1], dim=-1)

        # Swap last two dimensions
        y = y.swapaxes(-1, -2)
        return y


class LayerDemapper(Block):
    r"""Demaps MIMO layers to coded transport block(s) by following Sec. 6.3.1.3
    and Sec. 7.3.1.3 in :cite:p:`3GPPTS38211`.

    This block must be associated to a :class:`~sionna.phy.nr.LayerMapper` and
    performs the inverse operation.

    It is assumed that ``num_bits_per_symbol`` consecutive LLRs belong to
    a single symbol position. This allows to apply the LayerDemapper after
    demapping symbols to LLR values.

    If the layer mapper is configured for dual codeword transmission, a list of
    both transport block streams is returned.

    :param layer_mapper: Associated :class:`~sionna.phy.nr.LayerMapper`.
    :param num_bits_per_symbol: Modulation order. Defines how many consecutive
        LLRs are associated to the same symbol position.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input inputs: [..., num_layers, n/num_layers], `torch.float`.
        MIMO layer data sequences.

    :output llr: [..., n] or [[..., n1], [..., n2]], `torch.float`.
        Sequence of bits after layer demapping.
        If ``num_codewords`` = 2, a list of two transport blocks is returned.

    .. rubric:: Notes

    As it is more convenient to apply the layer demapper after demapping
    symbols to LLRs, this block groups the input sequence into groups of
    ``num_bits_per_symbol`` LLRs before restoring the original symbol sequence.
    This behavior can be deactivated by setting ``num_bits_per_symbol`` = 1.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.nr import LayerMapper, LayerDemapper

        mapper = LayerMapper(num_layers=2)
        demapper = LayerDemapper(mapper, num_bits_per_symbol=4)

        symbols = torch.randn(10, 100, dtype=torch.complex64)
        mapped = mapper(symbols)
        # After channel + demapping to LLRs...
        llrs = torch.randn(10, 2, 200)  # num_bits_per_symbol=4, so 50*4=200
        demapped = demapper(llrs)
        print(demapped.shape)
        # torch.Size([10, 400])
    """

    def __init__(
        self,
        layer_mapper: LayerMapper,
        num_bits_per_symbol: int = 1,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(layer_mapper, LayerMapper):
            raise TypeError("layer_mapper must be LayerMapper.")
        self._mapper = layer_mapper

        if num_bits_per_symbol % 1 != 0:
            raise TypeError("num_bits_per_symbol must be int.")
        self._num_bits_per_symbol = int(num_bits_per_symbol)

    def build(self, input_shapes):
        """Test input shapes for consistency."""
        num_layers = self._mapper.num_layers
        if input_shapes[-2] != num_layers:
            raise ValueError(
                "Invalid input dimension: input shape must be [..., num_layers, n]."
            )

        if input_shapes[-1] % self._num_bits_per_symbol != 0:
            raise ValueError(
                "Invalid input dimension: last dimension must be a multiple of "
                "num_bits_per_symbol."
            )

    def call(
        self, inputs: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Demaps multiple layers back to transport block stream(s).

        :param inputs: MIMO layer data sequences.

        :output llr: Sequence(s) of bits after layer demapping.
        """
        # Group LLRs into blocks of num_bits_per_symbol values
        s = inputs.shape[-1]
        x = split_dim(
            inputs,
            (int(s / self._num_bits_per_symbol), self._num_bits_per_symbol),
            axis=len(inputs.shape) - 1,
        )

        # Swap last dimensions
        x = x.swapaxes(-2, -3)

        if self._mapper.num_codewords == 1:
            y = flatten_last_dims(x, num_dims=3)
            return y
        else:
            # Multiplex into two codewords/streams
            # Only relevant for PDSCH with dual codeword transmission
            y0 = flatten_last_dims(x[..., : self._mapper.num_layers0, :], num_dims=3)
            y1 = flatten_last_dims(x[..., self._mapper.num_layers0 :, :], num_dims=3)
            return [y0, y1]

