#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for Upsampling block"""

import torch

from sionna.phy import dtypes
from sionna.phy.signal import Upsampling


class TestUpsampling:
    """Tests for the Upsampling class"""

    def test_shape_default_axis(self, device, precision):
        """Test the output shape with default axis (-1)"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        expected_shape = [32, 100 * samples_per_symbol]
        assert list(y.shape) == expected_shape

    def test_shape_different_axis(self, device, precision):
        """Test the output shape with different axis values"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        # Test with axis=1 (upsampling the second dimension)
        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol,
            axis=1,
            precision=precision,
            device=device,
        )

        x = torch.randn(32, 50, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        expected_shape = [32, 50 * samples_per_symbol, 100]
        assert list(y.shape) == expected_shape

    def test_dtype_preservation_real(self, device, precision):
        """Test that real dtype is preserved"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        upsampler = Upsampling(samples_per_symbol=4, precision=precision, device=device)

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        assert y.dtype == rdtype

    def test_dtype_preservation_complex(self, device, precision):
        """Test that complex dtype is preserved"""
        cdtype = dtypes[precision]["torch"]["cdtype"]
        upsampler = Upsampling(samples_per_symbol=4, precision=precision, device=device)

        x = torch.randn(32, 100, dtype=cdtype, device=device)
        y = upsampler(x)

        assert y.dtype == cdtype

    def test_zero_insertion(self, device, precision):
        """Test that zeros are correctly inserted between samples"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 4

        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        # Create a simple input
        x = torch.tensor([1.0, 2.0, 3.0], dtype=rdtype, device=device)
        y = upsampler(x)

        # Expected: [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
        expected = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            dtype=rdtype,
            device=device,
        )

        assert torch.allclose(y, expected)

    def test_upsampling_factor_1(self, device, precision):
        """Test that upsampling factor of 1 returns the input unchanged"""
        rdtype = dtypes[precision]["torch"]["dtype"]

        upsampler = Upsampling(samples_per_symbol=1, precision=precision, device=device)

        x = torch.randn(32, 100, dtype=rdtype, device=device)
        y = upsampler(x)

        assert torch.allclose(y, x)

    def test_batched_input(self, device, precision):
        """Test with multi-dimensional batched input"""
        rdtype = dtypes[precision]["torch"]["dtype"]
        samples_per_symbol = 3

        upsampler = Upsampling(
            samples_per_symbol=samples_per_symbol, precision=precision, device=device
        )

        x = torch.randn(8, 16, 24, 50, dtype=rdtype, device=device)
        y = upsampler(x)

        expected_shape = [8, 16, 24, 50 * samples_per_symbol]
        assert list(y.shape) == expected_shape

    def test_gradient_flow(self, device):
        """Test that gradients flow through the upsampling operation"""
        rdtype = dtypes["double"]["torch"]["dtype"]

        upsampler = Upsampling(samples_per_symbol=4, precision="double", device=device)

        x = torch.randn(32, 100, dtype=rdtype, device=device, requires_grad=True)
        y = upsampler(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Due to zero-insertion, each input element contributes once to the output
        assert torch.allclose(x.grad, torch.ones_like(x))
