#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.turbo.encoding module."""

import os

import numpy as np
import pytest
import torch

from sionna.phy.fec.turbo import TurboEncoder
from sionna.phy.mapping import BinarySource


# Get test data directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(TEST_DIR, "..", "..", "codes", "turbo")


class TestTurboEncoder:
    """Tests for TurboEncoder class."""

    @pytest.mark.parametrize("terminate", [False, True])
    @pytest.mark.parametrize("k", [10, 20, 50, 100])
    @pytest.mark.parametrize("rate", [1 / 2, 1 / 3])
    def test_output_dim(self, rate, k, terminate, device):
        """Test with all-zero codeword that output dims are correct (=n)
        and output also equals all-zero.
        """
        bs = 10
        cl = 5  # constraint length

        n = int(k / rate)
        if terminate:
            n += int(cl / rate)

        enc = TurboEncoder(
            rate=rate, constraint_length=cl, terminate=terminate, device=device
        )
        u = torch.zeros(bs, k, dtype=torch.float32, device=device)
        c = enc(u)

        # If no termination is used, the output must be k/r
        if terminate is False:
            assert c.shape[-1] == n

        # Verify that coderate is correct (allow small epsilon)
        assert abs(enc.coderate - k / c.shape[-1]) < 1e-6

        # Also check that all-zero input yields all-zero output
        c_hat = torch.zeros_like(c)
        assert torch.equal(c, c_hat)

        # Test that output dim can change (in eager mode)
        k_new = k + 1
        n_new = int(k_new / rate)
        u_new = torch.zeros(bs, k_new, device=device)
        c_new = enc(u_new)

        # If no termination is used, the output must be k/r
        if terminate is False:
            assert c_new.shape[-1] == n_new

        # Also check that all-zero input yields all-zero output
        c_hat_new = torch.zeros_like(c_new)
        assert torch.equal(c_new, c_hat_new)

        # Verify that coderate is correctly updated
        assert abs(enc.coderate - k_new / c_new.shape[-1]) < 1e-6

    def test_invalid_inputs(self):
        """Test with invalid rate values and invalid constraint lengths as
        input. Only rates [1/2, 1/3] and constraint lengths [3, 4, 5, 6]
        are accepted currently.
        """
        rate_invalid = [0.2, 0.45, 0.01]
        rate_valid = [1 / 3, 1 / 2]

        constraint_length_invalid = [2, 9, 0]
        constraint_length_valid = [3, 4, 5, 6]

        for rate in rate_valid:
            for mu in constraint_length_invalid:
                with pytest.raises(ValueError):
                    TurboEncoder(rate=rate, constraint_length=mu)

        for rate in rate_invalid:
            for mu in constraint_length_valid:
                with pytest.raises(ValueError):
                    TurboEncoder(rate=rate, constraint_length=mu)

        gmat = [["101", "111", "000"], ["000", "010", "011"]]
        with pytest.raises((ValueError, TypeError)):
            TurboEncoder(gen_poly=gmat)

    def test_polynomial_input(self, device):
        """Test that different formats of input polynomials are accepted and
        raises exceptions when the generator polynomials fail assertions.
        """
        bs = 10
        k = 100
        rate = 1 / 2
        n = int(k / rate)
        u = torch.zeros(bs, k, device=device)

        g1 = ["101", "111"]
        g2 = ("101", "111")

        for gen_poly in [g1, g2]:
            enc = TurboEncoder(
                gen_poly=gen_poly, rate=rate, terminate=False, device=device
            )
            c = enc(u)
            assert c.shape[-1] == n

            # Also check that all-zero input yields all-zero output
            c_hat = torch.zeros(bs, n, device=device)
            assert torch.equal(c, c_hat)

        # Test invalid polynomials
        gs_invalid = [
            (["1001", "111"], ValueError),  # Different lengths
            (["1001", 111], TypeError),  # Non-string
            (("1211", "1101"), ValueError),  # Non-binary chars
        ]

        for g, expected_error in gs_invalid:
            with pytest.raises(expected_error):
                TurboEncoder(gen_poly=g)

    @pytest.mark.parametrize("shape", [[4, 5, 5], []])
    def test_multi_dimensional(self, shape, device):
        """Test against arbitrary shapes."""
        k = 120
        n = 240  # rate must be 1/2 or 1/3

        source = BinarySource(device=device)
        enc = TurboEncoder(
            rate=k / n, constraint_length=5, terminate=False, device=device
        )

        s = shape.copy()
        bs = int(np.prod(s)) if s else 1
        b = source([bs, k])
        if s:
            s.append(k)
            b_res = b.reshape(s)
        else:
            b_res = b.reshape(k)

        # Encode 2D tensor
        c = enc(b)
        # Encode multi-D tensor
        c_res = enc(b_res)

        # Test that shape was preserved
        expected_shape = list(b_res.shape[:-1]) + [n]
        assert list(c_res.shape) == expected_shape

        # And reshape to 2D shape
        c_res_flat = c_res.reshape(bs, n) if s else c_res.reshape(1, n)
        # Both versions should yield same result
        assert torch.equal(c, c_res_flat)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 120

        source = BinarySource(device=device)
        enc = TurboEncoder(rate=0.5, constraint_length=6, device=device)

        b = source([1, 15, k])
        b_rep = b.repeat(bs, 1, 1)

        c = enc(b_rep)

        for i in range(bs):
            assert torch.equal(c[0, :, :], c[i, :, :])

    def test_dtypes_flexible(self, device):
        """Test that encoder supports variable dtypes and yields same result."""
        dt_supported = (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        )

        bs = 10
        k = 32

        source = BinarySource(device=device)

        enc_ref = TurboEncoder(
            rate=0.5, constraint_length=6, precision="single", device=device
        )

        u = source([bs, k])
        c_ref = enc_ref(u)

        for dt in dt_supported:
            enc = TurboEncoder(
                rate=0.5, constraint_length=6, precision="single", device=device
            )
            u_dt = u.to(dt)
            c = enc(u_dt)

            c_32 = c.to(torch.float32)
            assert torch.equal(c_ref, c_32)

    def test_torch_compile(self, device):
        """Test that torch.compile works and XLA-like compilation is supported."""
        bs = 10
        k = 100

        source = BinarySource(device=device)

        for t in [False, True]:
            enc = TurboEncoder(
                rate=0.5, constraint_length=6, terminate=t, device=device
            )

            compiled_enc = torch.compile(enc)

            # Test that for arbitrary input only 0,1 values are output
            u = source([bs, k])
            x = compiled_enc(u)

            # Execute twice
            x2 = compiled_enc(u)
            assert torch.equal(x, x2)

            # And change batch_size
            u = source([bs + 1, k])
            x = compiled_enc(u)

            # Test no batch dim
            u = source([k])
            x = compiled_enc(u)

    @pytest.mark.parametrize("k", [40, 112, 168, 432])
    def test_ref_implementation(self, k, device):
        """Test against pre-encoded codewords from reference implementation."""
        if not os.path.exists(REF_PATH):
            pytest.skip("Reference data not found")

        enc = TurboEncoder(
            rate=1 / 3, terminate=True, constraint_length=4, device=device
        )

        uref = np.load(os.path.join(REF_PATH, f"ref_k{k}_u.npy"))
        cref = np.load(os.path.join(REF_PATH, f"ref_k{k}_x.npy"))

        u = torch.tensor(uref, dtype=torch.float32, device=device)
        c = enc(u)

        assert np.array_equal(c.cpu().numpy(), cref)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        encoder = TurboEncoder(rate=1 / 3, constraint_length=4, terminate=True)
        u = torch.randint(0, 2, (10, 40), dtype=torch.float32)
        c = encoder(u)
        # For k=40, rate=1/3, terminate=True, n = 40*3 + 12 = 132
        assert c.shape == torch.Size([10, 132])

