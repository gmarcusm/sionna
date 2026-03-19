#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the flat fading channel models"""

import torch

from sionna.phy.channel import (
    ApplyFlatFadingChannel,
    FlatFadingChannel,
    GenerateFlatFadingChannel,
    KroneckerModel,
    exp_corr_mat,
)
from sionna.phy.mapping import QAMSource


class TestGenerateFlatFadingChannel:
    """Tests for the GenerateFlatFadingChannel class"""

    def test_output_shape_and_dtype(self, device):
        """Verify output shape and dtype for single and double precision"""
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 128

        # Single precision
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, device=device)
        h = gen_chn(batch_size)
        assert h.shape == torch.Size([batch_size, num_rx_ant, num_tx_ant])
        assert h.dtype == torch.complex64

        # Double precision
        gen_chn = GenerateFlatFadingChannel(
            num_tx_ant, num_rx_ant, precision="double", device=device
        )
        h = gen_chn(batch_size)
        assert h.dtype == torch.complex128

    def test_with_spatial_correlation(self, device):
        """Verify that correlation matrices are correctly applied"""
        num_tx_ant = 4
        num_rx_ant = 16
        precision = "double"
        r_tx = exp_corr_mat(0.4, num_tx_ant, precision=precision, device=device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(
            num_tx_ant, num_rx_ant, spatial_corr=kron, precision=precision, device=device
        )

        def func():
            h = gen_chn(1000000)
            r_tx_hat = (h.mH @ h).mean(dim=0)
            r_rx_hat = (h @ h.mH).mean(dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for _ in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0] / iterations / num_rx_ant
            r_rx_hat += tmp[1] / iterations / num_tx_ant
        assert torch.allclose(r_tx, r_tx_hat, atol=1e-3)
        assert torch.allclose(r_rx, r_rx_hat, atol=1e-3)

    def test_property_setter(self, device):
        """Verify that spatial_corr can be set dynamically"""
        num_tx_ant = 4
        num_rx_ant = 16
        precision = "double"
        r_tx = exp_corr_mat(0.4, num_tx_ant, precision=precision, device=device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, precision=precision, device=device)

        def func():
            gen_chn.spatial_corr = kron
            h = gen_chn(1000000)
            r_tx_hat = (h.mH @ h).mean(dim=0)
            r_rx_hat = (h @ h.mH).mean(dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for _ in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0] / iterations / num_rx_ant
            r_rx_hat += tmp[1] / iterations / num_tx_ant
        assert torch.allclose(r_tx, r_tx_hat, atol=1e-3)
        assert torch.allclose(r_rx, r_rx_hat, atol=1e-3)

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        gen_chn = GenerateFlatFadingChannel(num_tx_ant=4, num_rx_ant=16, device=device)
        h = gen_chn(batch_size=32)
        assert h.shape == torch.Size([32, 16, 4])


class TestApplyFlatFadingChannel:
    """Tests for the ApplyFlatFadingChannel class"""

    def test_without_noise(self, device):
        """Verify y == h @ x when no noise is added"""
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 24
        r_tx = exp_corr_mat(0.4, num_tx_ant, device=device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, device=device)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(
            num_tx_ant, num_rx_ant, spatial_corr=kron, device=device
        )
        app_chn = ApplyFlatFadingChannel(device=device)
        h = gen_chn(batch_size)
        x = QAMSource(4, device=device)([batch_size, num_tx_ant])
        y = app_chn(x, h)
        expected = (h @ x.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(y, expected)

    def test_with_noise(self, device):
        """Verify that noise variance matches specified no"""
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 100000
        r_tx = exp_corr_mat(0.4, num_tx_ant, device=device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, device=device)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(
            num_tx_ant, num_rx_ant, spatial_corr=kron, device=device
        )
        app_chn = ApplyFlatFadingChannel(device=device)
        h = gen_chn(batch_size)
        x = QAMSource(4, device=device)([batch_size, num_tx_ant])
        no = 0.1
        y = app_chn(x, h, no)
        expected = (h @ x.unsqueeze(-1)).squeeze(-1)
        n = y - expected
        noise_var = n.var().item()
        assert abs(no - noise_var) < 1e-3

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        app_chn = ApplyFlatFadingChannel(device=device)
        x = torch.randn(32, 4, dtype=torch.complex64, device=device)
        h = torch.randn(32, 16, 4, dtype=torch.complex64, device=device)
        y = app_chn(x, h)
        assert y.shape == torch.Size([32, 16])


class TestFlatFadingChannel:
    """Tests for the FlatFadingChannel class"""

    def test_without_noise(self, device):
        """Verify channel output without noise matches expected"""
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 24
        precision = "double"
        r_tx = exp_corr_mat(0.4, num_tx_ant, precision=precision, device=device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(
            num_tx_ant,
            num_rx_ant,
            spatial_corr=kron,
            return_channel=True,
            precision=precision,
            device=device,
        )
        x = QAMSource(4, precision=precision, device=device)([batch_size, num_tx_ant])
        y, h = chn(x)
        expected = (h @ x.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(y, expected)

    def test_with_noise(self, device):
        """Verify that noise variance matches specified no"""
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 100000
        precision = "double"
        r_tx = exp_corr_mat(0.4, num_tx_ant, precision=precision, device=device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(
            num_tx_ant,
            num_rx_ant,
            spatial_corr=kron,
            return_channel=True,
            precision=precision,
            device=device,
        )
        x = QAMSource(4, precision=precision, device=device)([batch_size, num_tx_ant])
        no = 0.2
        y, h = chn(x, no)
        expected = (h @ x.unsqueeze(-1)).squeeze(-1)
        n = y - expected
        noise_var = n.var().item()
        assert abs(no - noise_var) < 1e-3

    def test_no_return_channel(self, device):
        """Verify output variance when return_channel is False"""
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 1000000
        chn = FlatFadingChannel(
            num_tx_ant, num_rx_ant, return_channel=False, device=device
        )
        x = QAMSource(4, device=device)([batch_size, num_tx_ant])
        no = 0.2
        y = chn(x, no)
        y_var = y.var().item()
        # The variance should be approximately num_tx_ant (signal power) + no (noise)
        assert abs(y_var - (num_tx_ant + no)) < 0.1

    def test_property_setter(self, device):
        """Verify that spatial_corr can be set dynamically on FlatFadingChannel"""
        num_tx_ant = 4
        num_rx_ant = 16
        precision = "double"
        r_tx = exp_corr_mat(0.4, num_tx_ant, precision=precision, device=device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(
            num_tx_ant, num_rx_ant, return_channel=True, precision=precision, device=device
        )
        qam_source = QAMSource(4, precision=precision, device=device)

        def func():
            chn.spatial_corr = kron
            x = qam_source([1000000, num_tx_ant])
            no = 0.2
            y, h = chn(x, no)
            r_tx_hat = (h.mH @ h).mean(dim=0)
            r_rx_hat = (h @ h.mH).mean(dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for _ in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0] / iterations / num_rx_ant
            r_rx_hat += tmp[1] / iterations / num_tx_ant
        assert torch.allclose(r_tx, r_tx_hat, atol=1e-3)
        assert torch.allclose(r_rx, r_rx_hat, atol=1e-3)

    def test_generate_property(self, device):
        """Verify access to internal GenerateFlatFadingChannel via generate property"""
        num_tx_ant = 4
        num_rx_ant = 16
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, device=device)
        assert isinstance(chn.generate, GenerateFlatFadingChannel)
        h = chn.generate(32)
        assert h.shape == torch.Size([32, num_rx_ant, num_tx_ant])

    def test_apply_property(self, device):
        """Verify access to internal ApplyFlatFadingChannel via apply property"""
        num_tx_ant = 4
        num_rx_ant = 16
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, device=device)
        assert isinstance(chn.apply, ApplyFlatFadingChannel)
        x = torch.randn(32, num_tx_ant, dtype=torch.complex64, device=device)
        h = torch.randn(32, num_rx_ant, num_tx_ant, dtype=torch.complex64, device=device)
        y = chn.apply(x, h)
        assert y.shape == torch.Size([32, num_rx_ant])

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        chn = FlatFadingChannel(
            num_tx_ant=4, num_rx_ant=16, return_channel=True, device=device
        )
        x = torch.randn(32, 4, dtype=torch.complex64, device=device)
        y, h = chn(x)
        assert y.shape == torch.Size([32, 16])
        assert h.shape == torch.Size([32, 16, 4])


class TestFlatFadingChannelCompiled:
    """Tests for the flat fading channel with torch.compile"""

    def test_compiled_generate(self, device):
        """Verify GenerateFlatFadingChannel works with torch.compile"""
        num_tx_ant = 4
        num_rx_ant = 16
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, device=device)

        @torch.compile
        def func(batch_size):
            return gen_chn(batch_size)

        h = func(32)
        assert h.shape == torch.Size([32, num_rx_ant, num_tx_ant])

    def test_compiled_apply(self, device):
        """Verify ApplyFlatFadingChannel works with torch.compile"""
        num_tx_ant = 4
        num_rx_ant = 16
        app_chn = ApplyFlatFadingChannel(device=device)

        @torch.compile
        def func(x, h, no):
            return app_chn(x, h, no)

        x = torch.randn(32, num_tx_ant, dtype=torch.complex64, device=device)
        h = torch.randn(32, num_rx_ant, num_tx_ant, dtype=torch.complex64, device=device)
        y = func(x, h, 0.1)
        assert y.shape == torch.Size([32, num_rx_ant])

    def test_compiled_flat_fading_channel(self, device):
        """Verify FlatFadingChannel works with torch.compile"""
        num_tx_ant = 4
        num_rx_ant = 16
        chn = FlatFadingChannel(
            num_tx_ant, num_rx_ant, return_channel=True, device=device
        )

        @torch.compile
        def func(x, no):
            return chn(x, no)

        x = torch.randn(32, num_tx_ant, dtype=torch.complex64, device=device)
        y, h = func(x, 0.1)
        assert y.shape == torch.Size([32, num_rx_ant])
        assert h.shape == torch.Size([32, num_rx_ant, num_tx_ant])

