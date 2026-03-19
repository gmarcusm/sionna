#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.ofdm.detection module."""

import pytest
import numpy as np
import torch

from sionna.phy.ofdm import (
    ResourceGrid,
    OFDMDetector,
    MaximumLikelihoodDetector,
    LinearDetector,
    KBestDetector,
    EPDetector,
    MMSEPICDetector,
)
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import complex_normal


@pytest.fixture
def resource_grid():
    """Create a simple resource grid for testing."""
    return ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=64,
        subcarrier_spacing=30e3,
        num_tx=2,
        num_streams_per_tx=2,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
    )


@pytest.fixture
def stream_management():
    """Create a StreamManagement for testing."""
    rx_tx_association = np.ones([1, 2])
    return StreamManagement(rx_tx_association, num_streams_per_tx=2)


class TestLinearDetector:
    """Tests for LinearDetector class."""

    @pytest.mark.parametrize("equalizer", ["lmmse", "zf", "mf"])
    @pytest.mark.parametrize("output", ["bit", "symbol"])
    def test_output_shape_bit_mode(
        self, device, precision, resource_grid, stream_management, equalizer, output
    ):
        """Test that LinearDetector produces correct output shape."""
        detector = LinearDetector(
            equalizer=equalizer,
            output=output,
            demapping_method="app",
            resource_grid=resource_grid,
            stream_management=stream_management,
            constellation_type="qam",
            num_bits_per_symbol=4,
            hard_out=False,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        # Create input tensors
        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        # Run detector
        llr = detector(y, h_hat, err_var, no)

        # Check output shape
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        if output == "bit":
            expected_shape = (
                batch_size,
                num_tx,
                num_streams_per_tx,
                num_data_symbols * 4,
            )  # 4 bits per symbol
        else:
            # For soft symbol output (logits over constellation points)
            expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols, 16)  # 2^4 = 16 points
        assert llr.shape == expected_shape

    @pytest.mark.parametrize("equalizer", ["lmmse", "zf", "mf"])
    def test_scalar_err_var(
        self, device, precision, resource_grid, stream_management, equalizer
    ):
        """Test that detector works with scalar float err_var (perfect CSI case)."""
        detector = LinearDetector(
            equalizer=equalizer,
            output="bit",
            demapping_method="app",
            resource_grid=resource_grid,
            stream_management=stream_management,
            constellation_type="qam",
            num_bits_per_symbol=4,
            hard_out=False,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        # Use scalar float for err_var (simulates perfect CSI case)
        err_var = 0.0
        no = torch.ones(1, device=device) * 0.1

        # Should not raise AttributeError
        llr = detector(y, h_hat, err_var, no)

        # Check output shape
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        expected_shape = (
            batch_size,
            num_tx,
            num_streams_per_tx,
            num_data_symbols * 4,
        )
        assert llr.shape == expected_shape

    @pytest.mark.parametrize("equalizer", ["lmmse", "zf", "mf"])
    def test_hard_decision(
        self, device, precision, resource_grid, stream_management, equalizer
    ):
        """Test that hard_out=True produces hard decisions."""
        detector = LinearDetector(
            equalizer=equalizer,
            output="bit",
            demapping_method="app",
            resource_grid=resource_grid,
            stream_management=stream_management,
            constellation_type="qam",
            num_bits_per_symbol=4,
            hard_out=True,
            precision=precision,
            device=device,
        )

        batch_size = 4
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        bits = detector(y, h_hat, err_var, no)

        # Hard decisions should be 0 or 1
        assert torch.all((bits == 0) | (bits == 1))


class TestMaximumLikelihoodDetector:
    """Tests for MaximumLikelihoodDetector class."""

    @pytest.mark.parametrize("output", ["bit", "symbol"])
    @pytest.mark.parametrize("demapping_method", ["app", "maxlog"])
    def test_output_shape(
        self, device, precision, output, demapping_method
    ):
        """Test that ML detector produces correct output shape."""
        # Use smaller dimensions for ML detector (computationally expensive)
        rg = ResourceGrid(
            num_ofdm_symbols=4,
            fft_size=16,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=2,  # 2 streams for ML detection
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[1],
        )
        sm = StreamManagement(np.ones([1, 1]), num_streams_per_tx=2)

        detector = MaximumLikelihoodDetector(
            output=output,
            demapping_method=demapping_method,
            resource_grid=rg,
            stream_management=sm,
            constellation_type="qam",
            num_bits_per_symbol=2,  # QPSK for faster test
            hard_out=False,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size
        num_tx = rg.num_tx
        num_streams_per_tx = rg.num_streams_per_tx
        num_effective_subcarriers = rg.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        result = detector(y, h_hat, err_var, no)

        num_data_symbols = rg.pilot_pattern.num_data_symbols
        if output == "bit":
            expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols * 2)
        else:
            expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols, 4)  # 2^2 = 4 points
        assert result.shape == expected_shape


class TestKBestDetector:
    """Tests for KBestDetector class."""

    def test_output_shape(self, device, precision):
        """Test that K-Best detector produces correct output shape."""
        rg = ResourceGrid(
            num_ofdm_symbols=4,
            fft_size=16,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=2,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[1],
        )
        sm = StreamManagement(np.ones([1, 1]), num_streams_per_tx=2)

        detector = KBestDetector(
            output="bit",
            num_streams=2,
            k=4,
            resource_grid=rg,
            stream_management=sm,
            constellation_type="qam",
            num_bits_per_symbol=2,
            hard_out=False,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size
        num_tx = rg.num_tx
        num_streams_per_tx = rg.num_streams_per_tx
        num_effective_subcarriers = rg.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        llr = detector(y, h_hat, err_var, no)

        num_data_symbols = rg.pilot_pattern.num_data_symbols
        expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols * 2)
        assert llr.shape == expected_shape


class TestEPDetector:
    """Tests for EPDetector class."""

    def test_output_shape(self, device, precision):
        """Test that EP detector produces correct output shape."""
        rg = ResourceGrid(
            num_ofdm_symbols=4,
            fft_size=16,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=2,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[1],
        )
        sm = StreamManagement(np.ones([1, 1]), num_streams_per_tx=2)

        detector = EPDetector(
            output="bit",
            resource_grid=rg,
            stream_management=sm,
            num_bits_per_symbol=2,
            hard_out=False,
            l=5,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size
        num_tx = rg.num_tx
        num_streams_per_tx = rg.num_streams_per_tx
        num_effective_subcarriers = rg.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        llr = detector(y, h_hat, err_var, no)

        num_data_symbols = rg.pilot_pattern.num_data_symbols
        expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols * 2)
        assert llr.shape == expected_shape


class TestMMSEPICDetector:
    """Tests for MMSEPICDetector class."""

    def test_output_shape(self, device, precision):
        """Test that MMSE PIC detector produces correct output shape."""
        rg = ResourceGrid(
            num_ofdm_symbols=4,
            fft_size=16,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=2,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[1],
        )
        sm = StreamManagement(np.ones([1, 1]), num_streams_per_tx=2)

        detector = MMSEPICDetector(
            output="bit",
            demapping_method="app",
            resource_grid=rg,
            stream_management=sm,
            num_iter=2,
            constellation_type="qam",
            num_bits_per_symbol=2,
            hard_out=False,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = rg.num_ofdm_symbols
        fft_size = rg.fft_size
        num_tx = rg.num_tx
        num_streams_per_tx = rg.num_streams_per_tx
        num_effective_subcarriers = rg.num_effective_subcarriers
        num_data_symbols = rg.pilot_pattern.num_data_symbols

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        # Prior: [batch_size, num_tx, num_streams, num_data_symbols * num_bits_per_symbol]
        prior = torch.zeros(
            (batch_size, num_tx, num_streams_per_tx, num_data_symbols * 2),
            dtype=torch.float32 if precision == "single" else torch.float64,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        llr = detector(y, h_hat, prior, err_var, no)

        expected_shape = (batch_size, num_tx, num_streams_per_tx, num_data_symbols * 2)
        assert llr.shape == expected_shape


class TestDetectorCompilation:
    """Tests for torch.compile compatibility."""

    def test_linear_detector_compiles(self, device, precision, resource_grid, stream_management):
        """Test that LinearDetector can be compiled with torch.compile."""
        if device == "cpu":
            pytest.skip("Compilation tests may be slow on CPU")

        detector = LinearDetector(
            equalizer="lmmse",
            output="bit",
            demapping_method="app",
            resource_grid=resource_grid,
            stream_management=stream_management,
            constellation_type="qam",
            num_bits_per_symbol=4,
            hard_out=False,
            precision=precision,
            device=device,
        )

        batch_size = 2
        num_rx = 1
        num_rx_ant = 4
        num_ofdm_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_tx = resource_grid.num_tx
        num_streams_per_tx = resource_grid.num_streams_per_tx
        num_effective_subcarriers = resource_grid.num_effective_subcarriers

        y = complex_normal(
            (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size),
            precision=precision,
            device=device,
        )
        h_hat = complex_normal(
            (
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_streams_per_tx,
                num_ofdm_symbols,
                num_effective_subcarriers,
            ),
            precision=precision,
            device=device,
        )
        err_var = torch.ones(1, device=device) * 0.01
        no = torch.ones(1, device=device) * 0.1

        # Compile the detector
        compiled_detector = torch.compile(detector)

        # Run compiled version
        llr_compiled = compiled_detector(y, h_hat, err_var, no)

        # Run original version
        llr_orig = detector(y, h_hat, err_var, no)

        # Results should match
        assert torch.allclose(llr_compiled, llr_orig, atol=1e-4)

