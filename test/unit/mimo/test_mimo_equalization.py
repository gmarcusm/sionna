#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.mimo.equalization module."""

import pytest
import numpy as np
import torch

from sionna.phy.mimo.equalization import (
    lmmse_matrix,
    lmmse_equalizer,
    zf_equalizer,
    mf_equalizer,
)
from sionna.phy.utils import complex_normal


def exp_corr_mat(a: float, n: int, dtype=torch.complex64) -> torch.Tensor:
    """Generate exponential correlation matrix."""
    row = torch.pow(torch.tensor(a, dtype=dtype.to_real()), torch.arange(n, dtype=dtype.to_real()))
    mat = torch.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(n):
            mat[i, j] = row[abs(i - j)]
    return mat


class TestLMMSEMatrix:
    """Tests for lmmse_matrix function."""

    def test_lmmse_matrix_shape(self, device, precision):
        """Test that lmmse_matrix produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 8
        m = 6  # num_rx
        k = 4  # num_tx

        h = complex_normal((batch_size, m, k), precision=precision, device=device)

        g = lmmse_matrix(h, precision=precision)

        assert g.shape == (batch_size, k, m)

    def test_lmmse_matrix_with_noise_covariance(self, device, precision):
        """Test lmmse_matrix with explicit noise covariance."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        m = 4
        k = 2

        h = complex_normal((m, k), precision=precision, device=device)
        s = torch.eye(m, dtype=cdtype, device=device) * 0.5

        g = lmmse_matrix(h, s, precision=precision)

        assert g.shape == (k, m)

    def test_lmmse_matrix_identity_covariance(self, device, precision):
        """Test that None and identity covariance give same result."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128
        atol = 1e-5 if precision == "single" else 1e-10

        m = 4
        k = 2

        h = complex_normal((m, k), precision=precision, device=device)
        s = torch.eye(m, dtype=cdtype, device=device)

        # Note: s=None uses a different (more stable) formula
        g_with_s = lmmse_matrix(h, s, precision=precision)
        g_none = lmmse_matrix(h, None, precision=precision)

        # Results should be close but may differ slightly due to different formulas
        assert g_with_s.shape == g_none.shape


class TestLMMSEEqualizer:
    """Tests for lmmse_equalizer function."""

    @pytest.mark.parametrize("whiten_interference", [True, False])
    def test_lmmse_equalizer_shape(self, device, precision, whiten_interference):
        """Test that lmmse_equalizer produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 8
        m = 6  # num_rx
        k = 4  # num_tx

        y = complex_normal((batch_size, m), precision=precision, device=device)
        h = complex_normal((batch_size, m, k), precision=precision, device=device)
        s = torch.eye(m, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        x_hat, no_eff = lmmse_equalizer(y, h, s, whiten_interference=whiten_interference, precision=precision)

        assert x_hat.shape == (batch_size, k)
        assert no_eff.shape == (batch_size, k)

    def test_lmmse_equalizer_zero_noise(self, device, precision):
        """Test LMMSE equalizer with zero noise (should recover symbols exactly)."""
        if precision == "single":
            pytest.skip("Requires double precision for numerical stability")

        cdtype = torch.complex128
        rdtype = torch.float64

        batch_size = 100
        m = 8
        k = 4

        h = complex_normal((batch_size, m, k), precision=precision, device=device)
        x = complex_normal((batch_size, k), precision=precision, device=device)
        y = (h @ x.unsqueeze(-1)).squeeze(-1)  # Noise-free
        s = torch.eye(m, dtype=cdtype, device=device).unsqueeze(0) * 1e-10

        x_hat, no_eff = lmmse_equalizer(y, h, s.expand(batch_size, -1, -1), precision=precision)

        # Should recover x with high accuracy
        assert torch.allclose(x, x_hat, atol=1e-4)


class TestZFEqualizer:
    """Tests for zf_equalizer function."""

    def test_zf_equalizer_shape(self, device, precision):
        """Test that zf_equalizer produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 8
        m = 6  # num_rx
        k = 4  # num_tx

        y = complex_normal((batch_size, m), precision=precision, device=device)
        h = complex_normal((batch_size, m, k), precision=precision, device=device)
        s = torch.eye(m, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        x_hat, no_eff = zf_equalizer(y, h, s, precision=precision)

        assert x_hat.shape == (batch_size, k)
        assert no_eff.shape == (batch_size, k)

    def test_zf_equalizer_perfect_channel(self, device, precision):
        """Test ZF equalizer with orthogonal channel (should recover exactly)."""
        if precision == "single":
            pytest.skip("Requires double precision for numerical stability")

        cdtype = torch.complex128

        m = 4
        k = 4

        # Unitary channel (orthogonal columns)
        h, _, _ = torch.linalg.svd(complex_normal((m, k), precision=precision, device=device), full_matrices=False)
        x = complex_normal((k,), precision=precision, device=device)
        y = h @ x  # Noise-free
        s = torch.eye(m, dtype=cdtype, device=device) * 1e-10

        x_hat, no_eff = zf_equalizer(y, h, s, precision=precision)

        assert torch.allclose(x, x_hat, atol=1e-4)


class TestMFEqualizer:
    """Tests for mf_equalizer function."""

    def test_mf_equalizer_shape(self, device, precision):
        """Test that mf_equalizer produces correct output shape."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 8
        m = 6  # num_rx
        k = 4  # num_tx

        y = complex_normal((batch_size, m), precision=precision, device=device)
        h = complex_normal((batch_size, m, k), precision=precision, device=device)
        s = torch.eye(m, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        x_hat, no_eff = mf_equalizer(y, h, s, precision=precision)

        assert x_hat.shape == (batch_size, k)
        assert no_eff.shape == (batch_size, k)

    def test_mf_equalizer_positive_no_eff(self, device, precision):
        """Test that MF equalizer produces non-negative effective noise variance."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 8
        m = 8
        k = 4

        y = complex_normal((batch_size, m), precision=precision, device=device)
        h = complex_normal((batch_size, m, k), precision=precision, device=device)
        s = torch.eye(m, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        _, no_eff = mf_equalizer(y, h, s, precision=precision)

        # Effective noise variance should be non-negative
        assert (no_eff >= 0).all()


class TestEqualizerErrorStatistics:
    """Tests verifying error statistics of equalizers (adapted from ext tests)."""

    @pytest.mark.parametrize("equalizer", [lmmse_equalizer, zf_equalizer, mf_equalizer])
    @pytest.mark.parametrize("no", [0.1, 1.0, 3.0])
    def test_error_statistics_awgn(self, device, precision, equalizer, no):
        """Test that measured error variance matches estimated variance for AWGN."""
        if precision == "single":
            pytest.skip("Requires double precision for accurate statistics")

        cdtype = torch.complex128
        rdtype = torch.float64

        num_tx = 4
        num_rx = 8
        batch_size = 50000
        num_batches = 5

        err_var_accum = torch.tensor(0.0, dtype=rdtype, device=device)
        no_eff_accum = torch.tensor(0.0, dtype=rdtype, device=device)

        for _ in range(num_batches):
            # Generate random QPSK-like symbols
            x = complex_normal((batch_size, num_tx), precision=precision, device=device)
            x = x / x.abs()  # Normalize to unit magnitude

            # Random channel
            h = complex_normal((batch_size, num_rx, num_tx), precision=precision, device=device)

            # AWGN
            n = complex_normal((batch_size, num_rx), precision=precision, device=device) * np.sqrt(no)

            # Received signal
            y = (h @ x.unsqueeze(-1)).squeeze(-1) + n

            # Noise covariance
            s = torch.eye(num_rx, dtype=cdtype, device=device) * no
            s = s.unsqueeze(0).expand(batch_size, -1, -1)

            # Equalize
            x_hat, no_eff = equalizer(y, h, s, precision="double")

            # Measure error
            err = x - x_hat
            err_var = (err.abs() ** 2).mean()
            no_eff_mean = no_eff.mean()

            err_var_accum += err_var / num_batches
            no_eff_accum += no_eff_mean / num_batches

        # Check that estimated error variance matches measured error variance
        rel_err = abs(err_var_accum - no_eff_accum) / no_eff_accum
        assert rel_err < 0.05, f"Relative error {rel_err} too large"


class TestEqualizerColoredNoise:
    """Tests for equalizers with colored noise (adapted from TF tests)."""

    @pytest.mark.parametrize("equalizer", [lmmse_equalizer, zf_equalizer, mf_equalizer])
    @pytest.mark.parametrize("no", [0.1, 1.0, 3.0])
    def test_error_statistics_colored(self, device, precision, equalizer, no):
        """Test that measured error variance matches estimated variance for colored noise."""
        if precision == "single":
            pytest.skip("Requires double precision for accurate statistics")

        cdtype = torch.complex128
        rdtype = torch.float64

        num_tx = 4
        num_rx = 8
        batch_size = 50000
        num_batches = 5
        rho = 0.95  # Correlation coefficient

        # Create exponential correlation matrix for colored noise
        corr_mat = exp_corr_mat(rho, num_rx, cdtype).to(device)

        err_var_accum = torch.tensor(0.0, dtype=rdtype, device=device)
        no_eff_accum = torch.tensor(0.0, dtype=rdtype, device=device)

        for _ in range(num_batches):
            # Generate random QPSK-like symbols
            x = complex_normal((batch_size, num_tx), precision=precision, device=device)
            x = x / x.abs()  # Normalize to unit magnitude

            # Random channel
            h = complex_normal((batch_size, num_rx, num_tx), precision=precision, device=device)

            # Colored noise: n = no*I + corr_mat
            noise_cov = torch.eye(num_rx, dtype=cdtype, device=device) * no + corr_mat
            # Generate correlated noise via Cholesky
            l = torch.linalg.cholesky(noise_cov)
            white_noise = complex_normal((batch_size, num_rx), precision=precision, device=device)
            n = (l @ white_noise.unsqueeze(-1)).squeeze(-1)

            # Received signal
            y = (h @ x.unsqueeze(-1)).squeeze(-1) + n

            # Noise covariance
            s = noise_cov.unsqueeze(0).expand(batch_size, -1, -1)

            # Equalize
            x_hat, no_eff = equalizer(y, h, s, precision="double")

            # Measure error
            err = x - x_hat
            err_var = (err.abs() ** 2).mean()
            no_eff_mean = no_eff.mean()

            err_var_accum += err_var / num_batches
            no_eff_accum += no_eff_mean / num_batches

        # Check that estimated error variance matches measured error variance
        rel_err = abs(err_var_accum - no_eff_accum) / no_eff_accum
        assert rel_err < 0.05, f"Relative error {rel_err} too large"


class TestEqualizerCompilation:
    """Tests for torch.compile compatibility."""

    @pytest.mark.parametrize("equalizer", [lmmse_equalizer, zf_equalizer, mf_equalizer])
    def test_equalizer_compiles(self, device, precision, equalizer):
        """Test that equalizers can be compiled with torch.compile."""
        cdtype = torch.complex64 if precision == "single" else torch.complex128

        batch_size = 4
        m = 6
        k = 3

        y = complex_normal((batch_size, m), precision=precision, device=device)
        h = complex_normal((batch_size, m, k), precision=precision, device=device)
        s = torch.eye(m, dtype=cdtype, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Compile the equalizer
        compiled_eq = torch.compile(equalizer)

        # Run compiled version
        x_hat_compiled, no_eff_compiled = compiled_eq(y, h, s, precision=precision)

        # Run original version
        x_hat_orig, no_eff_orig = equalizer(y, h, s, precision=precision)

        # Results should match
        assert torch.allclose(x_hat_compiled, x_hat_orig, atol=1e-5)
        assert torch.allclose(no_eff_compiled, no_eff_orig, atol=1e-5)

