#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the spatial correlation models"""

import torch

from sionna.phy import config
from sionna.phy.channel import (
    KroneckerModel,
    PerColumnModel,
    exp_corr_mat,
    one_ring_corr_mat,
)
from sionna.phy.utils import complex_normal


class TestKroneckerModel:
    """Tests for the KroneckerModel class"""

    def test_covariance(self, device):
        """Verify that output covariance matches the expected correlation matrices"""
        M = 16
        K = 4
        precision = "double"
        r_tx = exp_corr_mat(0.4, K, precision=precision, device=device)
        r_rx = exp_corr_mat(0.99, M, precision=precision, device=device)
        batch_size = 1000000
        kron = KroneckerModel(r_tx, r_rx, precision=precision, device=device)

        def func():
            h = complex_normal([batch_size, M, K], precision=precision, device=device)
            h = kron(h)
            r_tx_hat = (h.mH @ h).mean(dim=0)
            r_rx_hat = (h @ h.mH).mean(dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for _ in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0] / iterations / M
            r_rx_hat += tmp[1] / iterations / K
        assert torch.allclose(r_tx, r_tx_hat, atol=1e-3)
        assert torch.allclose(r_rx, r_rx_hat, atol=1e-3)

    def test_per_example_r_tx(self, device):
        """Configure a different tx correlation for each example"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 128
        a_tx = torch.from_numpy(config.np_rng.uniform(size=[batch_size])).to(device)
        r_tx = exp_corr_mat(a_tx, K, precision=precision, device=device)
        r_rx = exp_corr_mat(0.99, M, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx, precision=precision, device=device)
        h = complex_normal([batch_size, M, K], precision=precision, device=device)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = (
                torch.linalg.cholesky(r_rx) @ h[i] @ torch.linalg.cholesky(r_tx[i]).mH
            )
            assert torch.allclose(h_corr[i], h_test, atol=1e-10)

    def test_per_example_r_rx(self, device):
        """Configure a different rx correlation for each example"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        r_tx = exp_corr_mat(0.4, K, precision=precision, device=device)
        a_rx = torch.from_numpy(config.np_rng.uniform(size=[batch_size])).to(device)
        r_rx = exp_corr_mat(a_rx, M, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx, precision=precision, device=device)
        h = complex_normal([batch_size, M, K], precision=precision, device=device)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = (
                torch.linalg.cholesky(r_rx[i]) @ h[i] @ torch.linalg.cholesky(r_tx).mH
            )
            assert torch.allclose(h_corr[i], h_test)

    def test_per_example_corr(self, device):
        """Configure a different rx/tx correlation for each example"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        a_tx = torch.from_numpy(config.np_rng.uniform(size=[batch_size])).to(device)
        a_rx = torch.from_numpy(config.np_rng.uniform(size=[batch_size])).to(device)
        r_tx = exp_corr_mat(a_tx, K, precision=precision, device=device)
        r_rx = exp_corr_mat(a_rx, M, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx, precision=precision, device=device)
        h = complex_normal([batch_size, M, K], precision=precision, device=device)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = (
                torch.linalg.cholesky(r_rx[i])
                @ h[i]
                @ torch.linalg.cholesky(r_tx[i]).mH
            )
            assert torch.allclose(h_corr[i], h_test)

    def test_same_channel_with_different_corr(self, device):
        """Apply different correlation matrices to the same channel"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        a_tx = torch.from_numpy(config.np_rng.uniform(size=[batch_size])).to(device)
        a_rx = torch.from_numpy(config.np_rng.uniform(size=[batch_size])).to(device)
        r_tx = exp_corr_mat(a_tx, K, precision=precision, device=device)
        r_rx = exp_corr_mat(a_rx, M, precision=precision, device=device)
        kron = KroneckerModel(r_tx, r_rx, precision=precision, device=device)
        h = complex_normal([M, K], precision=precision, device=device)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = (
                torch.linalg.cholesky(r_rx[i]) @ h @ torch.linalg.cholesky(r_tx[i]).mH
            )
            assert torch.allclose(h_corr[i], h_test)

    def test_property_setter(self, device):
        """Check that correlation matrices can be changed"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 10
        kron = KroneckerModel(None, None)

        r_tx = exp_corr_mat(0.4, K, precision=precision, device=device)
        r_rx = exp_corr_mat(0.9, M, precision=precision, device=device)
        kron.r_tx = r_tx
        kron.r_rx = r_rx
        h = complex_normal([batch_size, M, K], precision=precision, device=device)
        h_corr = kron(h)

        for i in range(batch_size):
            h_test = torch.linalg.cholesky(r_rx) @ h[i] @ torch.linalg.cholesky(r_tx).mH
            assert torch.allclose(h_corr[i], h_test, atol=1e-6)

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        # Create correlation matrices
        r_tx = exp_corr_mat(0.4, 4, device=device)  # 4x4 TX correlation
        r_rx = exp_corr_mat(0.9, 16, device=device)  # 16x16 RX correlation

        # Create model
        kron = KroneckerModel(r_tx, r_rx)

        # Apply to channel matrix
        h = torch.randn(32, 16, 4, dtype=torch.complex64, device=device)
        h_corr = kron(h)
        assert h_corr.shape == torch.Size([32, 16, 4])


class TestPerColumnModel:
    """Tests for the PerColumnModel class"""

    def test_covariance(self, device):
        """Verify that output covariance matches the expected per-column correlation"""
        M = 16
        K = 4
        precision = "double"
        phi_deg = torch.tensor([-45.0, -15.0, 0.0, 30.0], device=device)
        r_rx = one_ring_corr_mat(phi_deg, M, precision=precision, device=device)
        batch_size = 100000
        onering = PerColumnModel(r_rx, precision=precision, device=device)

        def func():
            h = complex_normal([batch_size, M, K], precision=precision, device=device)
            h = onering(h)
            # Transpose to [K, batch_size, M]
            h = h.permute(2, 0, 1)
            # Add dimension: [K, batch_size, M, 1]
            h = h.unsqueeze(-1)
            # Compute covariance: [K, batch_size, M, M] -> [K, M, M]
            r_rx_hat = (h @ h.mH).mean(dim=1)
            return r_rx_hat

        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 100
        for _ in range(iterations):
            r_rx_hat += func() / iterations
        assert torch.allclose(r_rx, r_rx_hat, atol=1e-3)

    def test_per_example_corr(self, device):
        """Apply different per-example correlation matrices"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 24
        phi_deg = torch.from_numpy(config.np_rng.uniform(size=[batch_size, K])).to(
            device
        )
        r_rx = one_ring_corr_mat(phi_deg, M, precision=precision, device=device)
        onering = PerColumnModel(r_rx, precision=precision, device=device)

        h = complex_normal([batch_size, M, K], precision=precision, device=device)
        h_corr = onering(h)

        for i in range(batch_size):
            for k in range(K):
                h_test = torch.linalg.cholesky(r_rx[i, k]) @ h[i, :, k].unsqueeze(-1)
                h_test = h_test.squeeze(-1)
                assert torch.allclose(h_corr[i, :, k], h_test, atol=1e-10)

    def test_property_setter(self, device):
        """Check that the correlation matrix property can be changed"""
        M = 16
        K = 4
        precision = "double"
        batch_size = 24
        onering = PerColumnModel(None, precision=precision, device=device)

        h = complex_normal([batch_size, M, K], precision=precision, device=device)
        phi_deg = torch.empty(batch_size, K, device=device).uniform_(-70, 70)
        r_rx = one_ring_corr_mat(phi_deg, M, precision=precision, device=device)
        onering.r_rx = r_rx
        h_corr = onering(h)

        for i in range(batch_size):
            for k in range(K):
                h_test = torch.linalg.cholesky(r_rx[i, k]) @ h[i, :, k].unsqueeze(-1)
                h_test = h_test.squeeze(-1)
                assert torch.allclose(h_corr[i, :, k], h_test, atol=1e-6)

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        # Create per-column correlation matrices (4 users, 16 antennas)
        r_rx = one_ring_corr_mat(
            torch.tensor([-45.0, -15.0, 0.0, 30.0], device=device), 16, device=device
        )

        # Create model
        per_col = PerColumnModel(r_rx)

        # Apply to channel matrix
        h = torch.randn(32, 16, 4, dtype=torch.complex64, device=device)
        h_corr = per_col(h)
        assert h_corr.shape == torch.Size([32, 16, 4])
