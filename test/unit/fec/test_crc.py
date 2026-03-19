#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.crc.CRCEncoder and CRCDecoder."""

import os
import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
from sionna.phy.mapping import BinarySource


# Get reference data directory
test_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(test_dir, "..", "..", "codes", "crc")

VALID_POLS = ["CRC24A", "CRC24B", "CRC24C", "CRC16", "CRC11", "CRC6"]
PRECISION_DTYPES = [("single", torch.float32), ("double", torch.float64)]


class TestCRCEncoder:
    """Tests for the CRCEncoder class."""

    @pytest.mark.parametrize("idx,pol", enumerate(VALID_POLS))
    def test_polynomials(self, idx, pol):
        """Check that all valid polynomials from 38.212 are supported."""
        crc_polys = [
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1],
        ]

        c = CRCEncoder(pol)
        assert np.array_equal(c.crc_pol, crc_polys[idx])
        assert c.crc_length == (len(crc_polys[idx]) - 1)

    def test_invalid_inputs(self):
        """Test that invalid input raises an Exception."""
        # Non-string input
        with pytest.raises(TypeError):
            CRCEncoder(24)
        # Unknown CRC polynomial
        with pytest.raises(ValueError):
            CRCEncoder("CRC17")

    @pytest.mark.parametrize("pol", VALID_POLS)
    @pytest.mark.parametrize(
        "shape", [[10], [1, 10], [10, 3, 3], [1, 2, 3, 4, 100]]
    )
    def test_output_dim(self, device, pol, shape):
        """Test that output dims are correct (=k+crc_len)."""
        crc_enc = CRCEncoder(pol, device=device)
        crc_dec = CRCDecoder(crc_enc, device=device)

        u = torch.zeros(shape, device=device)
        c = crc_enc(u)
        u_hat, crc_indicator = crc_dec(c)

        # Output shapes are equal to input shape (besides last dim)
        assert list(c.shape[:-1]) == shape[:-1]
        # Last dimension of output is increased by 'crc_length'
        assert c.shape[-1] == shape[-1] + crc_enc.crc_length
        # Check dimensions of "crc_valid indicator" (boolean)
        assert list(crc_indicator.shape[:-1]) == shape[:-1]
        assert crc_indicator.shape[-1] == 1
        # Check that decoder removes parity bits (=original shape)
        assert list(u_hat.shape) == shape

    @pytest.mark.parametrize("shape", [[10], [2, 10], [1, 2, 3, 4, 100]])
    def test_torch_compile(self, device, shape):
        """Test that torch.compile works as expected."""
        pol = "CRC24A"

        crc_enc = CRCEncoder(pol, device=device)
        crc_dec = CRCDecoder(crc_enc, device=device)

        @torch.compile
        def run_crc(u):
            x = crc_enc(u)
            y, z = crc_dec(x)
            return y, z

        u = torch.zeros(shape, device=device)
        y, z = run_crc(u)
        assert y.shape == u.shape
        assert z.all()

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_torch_compile_modes(self, device, mode):
        """Test that torch.compile works with different compilation modes."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode can be slow on CPU")

        pol = "CRC24A"
        shape = [10, 100]

        crc_enc = CRCEncoder(pol, device=device)
        crc_dec = CRCDecoder(crc_enc, device=device)

        source = BinarySource(device=device)
        u = source(shape)

        # Reference without compilation
        c_ref = crc_enc(u)
        y_ref, z_ref = crc_dec(c_ref)

        # Compile and run with fresh input to avoid CUDA graph tensor reuse issues
        # Clone the input to create a new tensor that CUDA graphs won't conflict with
        u_compiled = u.clone()
        compiled_enc = torch.compile(crc_enc, mode=mode)
        compiled_dec = torch.compile(crc_dec, mode=mode)

        # For reduce-overhead mode with CUDA graphs, we need to mark step boundaries
        # when passing tensors between separately compiled functions
        if mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        c_compiled = compiled_enc(u_compiled)

        if mode == "reduce-overhead":
            # Clone to avoid CUDA graph tensor ownership conflicts between
            # separately compiled encoder and decoder
            c_compiled = c_compiled.clone()
            torch.compiler.cudagraph_mark_step_begin()
        y_compiled, z_compiled = compiled_dec(c_compiled)

        assert torch.equal(c_ref, c_compiled)
        assert torch.equal(y_ref, y_compiled)
        assert torch.equal(z_ref, z_compiled)

    @pytest.mark.parametrize("pol", VALID_POLS)
    @pytest.mark.parametrize(
        "shape", [[100], [100, 10], [4, 2, 100], [1, 100000]]
    )
    def test_valid_crc(self, device, pol, shape):
        """Test that CRC of error-free codewords always holds."""
        source = BinarySource(device=device)
        crc_enc = CRCEncoder(pol, device=device)
        crc_dec = CRCDecoder(crc_enc, device=device)

        u = source(shape)
        x = crc_enc(u)  # Add CRC parity bits
        _, crc_valid = crc_dec(x)  # Perform CRC check

        # CRC check for CRC encoded data x must always hold
        assert crc_valid.all()

    @pytest.mark.parametrize("pol", VALID_POLS)
    def test_error_patterns(self, device, pol):
        """Test that CRC detects random error patterns."""
        shape = [10, 100]
        source = BinarySource(device=device)

        crc_enc = CRCEncoder(pol, device=device)
        crc_dec = CRCDecoder(crc_enc, device=device)

        u = source(shape)
        x = crc_enc(u)  # Add CRC

        # For shorter CRCs (like CRC6), use fewer errors to ensure reliable
        # detection. CRC-6 has only 6 parity bits, so random 3-bit errors
        # have ~1/64 probability of being undetected.
        num_errors = 1 if crc_enc.crc_length <= 6 else 3

        # Add error patterns
        n_coded = x.shape[-1]
        e = torch.zeros_like(x)
        for i in range(x.shape[0]):
            error_pos = torch.randperm(n_coded, device=device)[:num_errors]
            e[i, error_pos] = 1

        # Add error vector
        x_err = torch.fmod(x + e, 2)

        _, crc_valid = crc_dec(x_err)  # Perform CRC check

        # CRC should detect all errors (= all checks return False)
        assert not crc_valid.any()

    @pytest.mark.parametrize("idx,pol", enumerate(VALID_POLS))
    def test_examples(self, idx, pol):
        """Test against some manually calculated examples."""
        crc_polys = [
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
        ]

        crc_length = len(crc_polys[idx])
        crc_enc = CRCEncoder(pol)
        u = torch.tensor([[1.0]], dtype=torch.float32)
        x = crc_enc(u)
        x = x.flatten()[-crc_length:].cpu().numpy()
        x_ref = np.array(crc_polys[idx])
        assert np.array_equal(x, x_ref)

    @pytest.mark.parametrize("pol", VALID_POLS)
    def test_valid_encoding(self, device, pol):
        """Check all valid polynomials from 38.212 against
        a dataset from a reference implementation."""
        if not os.path.exists(ref_path):
            pytest.skip("Reference data not found")

        # Load reference codewords
        u = np.load(os.path.join(ref_path, f"crc_u_{pol}.npy"))
        x_ref_np = np.load(os.path.join(ref_path, f"crc_x_ref_np_{pol}.npy"))

        crc_enc = CRCEncoder(pol, device=device)

        u_t = torch.tensor(u, dtype=torch.float32, device=device)
        x = crc_enc(u_t)  # Add CRC
        x_crc = x.flatten()[-crc_enc.crc_length :].cpu().numpy()

        assert np.array_equal(x_crc, x_ref_np)

        # Test properties k, n
        assert crc_enc.k == u.shape[-1]
        assert crc_enc.n == x.shape[-1]

    @pytest.mark.parametrize("p_in,dt_in", PRECISION_DTYPES)
    @pytest.mark.parametrize("p_enc,dt_enc", PRECISION_DTYPES)
    @pytest.mark.parametrize("p_dec,dt_dec", PRECISION_DTYPES)
    def test_dtype(self, device, p_in, dt_in, p_enc, dt_enc, p_dec, dt_dec):
        """Test support for variable dtypes."""
        pol = "CRC24A"
        shape = [2, 10]
        source = BinarySource(device=device)

        crc_enc = CRCEncoder(pol, precision=p_enc, device=device)
        crc_dec = CRCDecoder(crc_enc, precision=p_dec, device=device)

        u = source(shape).to(dt_in)
        x = crc_enc(u)
        y, _ = crc_dec(x)

        assert u.dtype == dt_in
        assert x.dtype == dt_enc
        assert y.dtype == dt_dec


class TestCRCDecoder:
    """Tests for the CRCDecoder class."""

    def test_invalid_encoder(self):
        """Test that decoder raises error for invalid encoder."""
        with pytest.raises(TypeError):
            CRCDecoder("not_an_encoder")

    def test_input_too_short(self, device):
        """Test that decoder raises error when input is too short."""
        enc = CRCEncoder("CRC24A", device=device)
        dec = CRCDecoder(enc, device=device)

        # Input shorter than CRC length
        with pytest.raises(ValueError):
            dec(torch.zeros(10, device=device))

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        encoder = CRCEncoder("CRC24A")
        decoder = CRCDecoder(encoder)

        bits = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        encoded = encoder(bits)
        decoded, crc_valid = decoder(encoded)

        assert decoded.shape == torch.Size([10, 100])
        assert crc_valid.all()

    @pytest.mark.parametrize("pol", VALID_POLS)
    def test_roundtrip(self, device, pol):
        """Test that encoder followed by decoder returns original bits."""
        source = BinarySource(device=device)
        shape = [10, 50]

        enc = CRCEncoder(pol, device=device)
        dec = CRCDecoder(enc, device=device)

        u = source(shape)
        c = enc(u)
        u_hat, crc_valid = dec(c)

        assert torch.equal(u, u_hat)
        assert crc_valid.all()

