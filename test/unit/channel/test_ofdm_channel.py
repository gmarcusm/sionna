#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the OFDM channel models"""

from typing import Tuple

import torch

from sionna.phy.channel import (
    ApplyOFDMChannel,
    GenerateOFDMChannel,
    OFDMChannel,
)
from sionna.phy.utils import complex_normal


class MockResourceGrid:
    """Mock resource grid for testing OFDM channel classes."""

    def __init__(
        self,
        num_ofdm_symbols: int = 14,
        fft_size: int = 64,
        subcarrier_spacing: float = 15e3,
        cyclic_prefix_length: int = 4,
    ):
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.subcarrier_spacing = subcarrier_spacing
        self.cyclic_prefix_length = cyclic_prefix_length

    @property
    def ofdm_symbol_duration(self) -> float:
        return (1 + self.cyclic_prefix_length / self.fft_size) / self.subcarrier_spacing


class MockChannelModel:
    """Mock channel model for testing OFDM channel classes.

    Generates random Rayleigh-like channel impulse responses.
    """

    def __init__(
        self,
        num_rx: int = 1,
        num_rx_ant: int = 2,
        num_tx: int = 1,
        num_tx_ant: int = 2,
        num_paths: int = 1,
        precision: str = "single",
        device: str = "cpu",
    ):
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_paths = num_paths
        self.precision = precision
        self.device = device

    def __call__(
        self, batch_size: int, num_time_steps: int, sampling_frequency: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate channel impulse response.

        Returns:
            h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
            tau: [batch_size, num_rx, num_tx, num_paths]
        """
        # Generate random path coefficients
        shape = [
            batch_size,
            self.num_rx,
            self.num_rx_ant,
            self.num_tx,
            self.num_tx_ant,
            self.num_paths,
            num_time_steps,
        ]
        h = complex_normal(shape, precision=self.precision, device=self.device)

        # Generate random delays (small values for single-tap like behavior)
        tau_shape = [batch_size, self.num_rx, self.num_tx, self.num_paths]
        dtype = torch.float32 if self.precision == "single" else torch.float64
        tau = torch.rand(tau_shape, dtype=dtype, device=self.device) * 1e-7

        return h, tau


class TestApplyOFDMChannel:
    """Tests for the ApplyOFDMChannel class"""

    def test_output_shape_and_dtype(self, device):
        """Verify output shape and dtype for single and double precision"""
        batch_size = 16
        num_tx, num_tx_ant = 2, 4
        num_rx, num_rx_ant = 1, 8
        num_ofdm_symbols, fft_size = 14, 64

        # Single precision
        apply_ch = ApplyOFDMChannel(device=device)
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        y = apply_ch(x, h_freq)
        assert y.shape == torch.Size(
            [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        )
        assert y.dtype == torch.complex64

        # Double precision
        apply_ch = ApplyOFDMChannel(precision="double", device=device)
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex128,
            device=device,
        )
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex128,
            device=device,
        )
        y = apply_ch(x, h_freq)
        assert y.dtype == torch.complex128

    def test_without_noise(self, device):
        """Verify y = sum over tx of h_freq * x when no noise is added"""
        batch_size = 8
        num_tx, num_tx_ant = 2, 2
        num_rx, num_rx_ant = 1, 4
        num_ofdm_symbols, fft_size = 7, 32

        apply_ch = ApplyOFDMChannel(precision="double", device=device)

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex128,
            device=device,
        )
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex128,
            device=device,
        )

        y = apply_ch(x, h_freq)

        # Manual computation: expand x and sum
        # x: [batch, num_tx, num_tx_ant, num_ofdm, fft] -> [batch, 1, 1, num_tx, num_tx_ant, num_ofdm, fft]
        x_expanded = x.unsqueeze(1).unsqueeze(1)
        expected = (h_freq * x_expanded).sum(dim=4).sum(dim=3)

        assert torch.allclose(y, expected)

    def test_with_noise(self, device):
        """Verify that noise variance matches specified no"""
        batch_size = 100000
        num_tx, num_tx_ant = 1, 1
        num_rx, num_rx_ant = 1, 1
        num_ofdm_symbols, fft_size = 14, 64
        no = 0.1

        apply_ch = ApplyOFDMChannel(device=device)

        x = torch.zeros(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        h_freq = torch.ones(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )

        y = apply_ch(x, h_freq, no)

        # y should be pure noise with variance no
        noise_var = y.var().item()
        assert abs(no - noise_var) < 1e-2

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        apply_ch = ApplyOFDMChannel(device=device)

        batch_size, num_tx, num_tx_ant = 16, 2, 4
        num_rx, num_rx_ant = 1, 8
        num_ofdm_symbols, fft_size = 14, 64

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )

        y = apply_ch(x, h_freq)
        assert y.shape == torch.Size([16, 1, 8, 14, 64])


class TestGenerateOFDMChannel:
    """Tests for the GenerateOFDMChannel class"""

    def test_output_shape_and_dtype(self, device):
        """Verify output shape and dtype for single and double precision"""
        batch_size = 32
        num_rx, num_rx_ant = 1, 2
        num_tx, num_tx_ant = 1, 4
        num_ofdm_symbols = 14
        fft_size = 64

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)

        # Single precision
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )
        gen_ch = GenerateOFDMChannel(channel_model, rg, device=device)
        h_freq = gen_ch(batch_size)

        assert h_freq.shape == torch.Size(
            [
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_tx_ant,
                num_ofdm_symbols,
                fft_size,
            ]
        )
        assert h_freq.dtype == torch.complex64

        # Double precision
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision="double",
            device=device,
        )
        gen_ch = GenerateOFDMChannel(
            channel_model, rg, precision="double", device=device
        )
        h_freq = gen_ch(batch_size)
        assert h_freq.dtype == torch.complex128

    def test_with_normalization(self, device):
        """Verify that normalization produces unit average energy"""
        batch_size = 1000
        num_rx, num_rx_ant = 1, 2
        num_tx, num_tx_ant = 1, 2
        num_ofdm_symbols = 14
        fft_size = 64

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            num_paths=4,
            device=device,
        )
        gen_ch = GenerateOFDMChannel(
            channel_model, rg, normalize_channel=True, device=device
        )
        h_freq = gen_ch(batch_size)

        # Compute average energy per resource element
        avg_energy = h_freq.abs().square().mean().item()
        # With normalization, average energy should be close to 1
        assert abs(avg_energy - 1.0) < 0.5  # Allow some tolerance due to randomness

    def test_different_resource_grid_params(self, device):
        """Verify that different resource grid parameters work correctly"""
        batch_size = 16
        num_rx, num_rx_ant = 2, 4
        num_tx, num_tx_ant = 2, 2

        # Test with various resource grid configurations
        for num_ofdm_symbols in [7, 14]:
            for fft_size in [32, 128]:
                rg = MockResourceGrid(
                    num_ofdm_symbols=num_ofdm_symbols,
                    fft_size=fft_size,
                    subcarrier_spacing=30e3,
                )
                channel_model = MockChannelModel(
                    num_rx=num_rx,
                    num_rx_ant=num_rx_ant,
                    num_tx=num_tx,
                    num_tx_ant=num_tx_ant,
                    device=device,
                )
                gen_ch = GenerateOFDMChannel(channel_model, rg, device=device)
                h_freq = gen_ch(batch_size)

                assert h_freq.shape == torch.Size(
                    [
                        batch_size,
                        num_rx,
                        num_rx_ant,
                        num_tx,
                        num_tx_ant,
                        num_ofdm_symbols,
                        fft_size,
                    ]
                )


class TestOFDMChannel:
    """Tests for the OFDMChannel class"""

    def test_output_shape_and_dtype(self, device):
        """Verify output shape and dtype for single and double precision"""
        batch_size = 16
        num_tx, num_tx_ant = 1, 4
        num_rx, num_rx_ant = 1, 2
        num_ofdm_symbols = 14
        fft_size = 64

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        # Single precision
        ofdm_ch = OFDMChannel(channel_model, rg, return_channel=True, device=device)
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        y, h_freq = ofdm_ch(x)

        assert y.shape == torch.Size(
            [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        )
        assert h_freq.shape == torch.Size(
            [
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_tx_ant,
                num_ofdm_symbols,
                fft_size,
            ]
        )
        assert y.dtype == torch.complex64

    def test_without_noise(self, device):
        """Verify channel output without noise matches expected"""
        batch_size = 8
        num_tx, num_tx_ant = 1, 2
        num_rx, num_rx_ant = 1, 2
        num_ofdm_symbols = 7
        fft_size = 32

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision="double",
            device=device,
        )

        ofdm_ch = OFDMChannel(
            channel_model, rg, return_channel=True, precision="double", device=device
        )
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex128,
            device=device,
        )
        y, h_freq = ofdm_ch(x)

        # Manual computation
        x_expanded = x.unsqueeze(1).unsqueeze(1)
        expected = (h_freq * x_expanded).sum(dim=4).sum(dim=3)

        assert torch.allclose(y, expected)

    def test_with_noise(self, device):
        """Verify that noise variance matches specified no"""
        batch_size = 100000
        num_tx, num_tx_ant = 1, 1
        num_rx, num_rx_ant = 1, 1
        num_ofdm_symbols = 14
        fft_size = 64
        no = 0.2

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        ofdm_ch = OFDMChannel(channel_model, rg, return_channel=True, device=device)
        x = torch.zeros(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        y, h_freq = ofdm_ch(x, no)

        # Signal through zero input is just noise
        noise_var = y.var().item()
        assert abs(no - noise_var) < 1e-2
        # Verify channel was returned
        assert h_freq is not None

    def test_no_return_channel(self, device):
        """Verify output when return_channel is False"""
        batch_size = 16
        num_tx, num_tx_ant = 1, 2
        num_rx, num_rx_ant = 1, 4
        num_ofdm_symbols = 14
        fft_size = 64

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        ofdm_ch = OFDMChannel(channel_model, rg, return_channel=False, device=device)
        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        y = ofdm_ch(x)

        # Should return only y, not a tuple
        assert isinstance(y, torch.Tensor)
        assert y.shape == torch.Size(
            [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        )

    def test_generate_property(self, device):
        """Verify access to internal GenerateOFDMChannel via generate property"""
        rg = MockResourceGrid()
        channel_model = MockChannelModel(device=device)

        ofdm_ch = OFDMChannel(channel_model, rg, device=device)
        assert isinstance(ofdm_ch.generate, GenerateOFDMChannel)
        h_freq = ofdm_ch.generate(32)
        assert h_freq.shape[0] == 32

    def test_apply_property(self, device):
        """Verify access to internal ApplyOFDMChannel via apply property"""
        batch_size = 16
        num_tx, num_tx_ant = 1, 2
        num_rx, num_rx_ant = 1, 2
        num_ofdm_symbols = 14
        fft_size = 64

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        ofdm_ch = OFDMChannel(channel_model, rg, device=device)
        assert isinstance(ofdm_ch.apply, ApplyOFDMChannel)

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        y = ofdm_ch.apply(x, h_freq)
        assert y.shape == torch.Size(
            [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        )


class TestOFDMChannelCompiled:
    """Tests for the OFDM channel with torch.compile"""

    def test_compiled_apply(self, device):
        """Verify ApplyOFDMChannel works with torch.compile"""
        batch_size = 16
        num_tx, num_tx_ant = 1, 2
        num_rx, num_rx_ant = 1, 2
        num_ofdm_symbols = 7
        fft_size = 32

        apply_ch = ApplyOFDMChannel(device=device)

        @torch.compile
        def func(x, h_freq, no):
            return apply_ch(x, h_freq, no)

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        y = func(x, h_freq, 0.1)
        assert y.shape == torch.Size(
            [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        )

    def test_compiled_generate(self, device):
        """Verify GenerateOFDMChannel works with torch.compile"""
        rg = MockResourceGrid()
        channel_model = MockChannelModel(device=device)
        gen_ch = GenerateOFDMChannel(channel_model, rg, device=device)

        @torch.compile
        def func(batch_size):
            return gen_ch(batch_size)

        h_freq = func(32)
        assert h_freq.shape[0] == 32

    def test_compiled_ofdm_channel(self, device):
        """Verify OFDMChannel works with torch.compile"""
        batch_size = 16
        num_tx, num_tx_ant = 1, 2
        num_rx, num_rx_ant = 1, 2
        num_ofdm_symbols = 14
        fft_size = 64

        rg = MockResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size)
        channel_model = MockChannelModel(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )
        ofdm_ch = OFDMChannel(channel_model, rg, return_channel=True, device=device)

        @torch.compile
        def func(x, no):
            return ofdm_ch(x, no)

        x = torch.randn(
            batch_size,
            num_tx,
            num_tx_ant,
            num_ofdm_symbols,
            fft_size,
            dtype=torch.complex64,
            device=device,
        )
        y, h_freq = func(x, 0.1)
        assert y.shape == torch.Size(
            [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        )
        assert h_freq.shape == torch.Size(
            [
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_tx_ant,
                num_ofdm_symbols,
                fft_size,
            ]
        )
