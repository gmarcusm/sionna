#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.linear.LinearEncoder."""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fec.utils import load_parity_check_examples
from sionna.phy.fec.linear import LinearEncoder
from sionna.phy.mapping import BinarySource


class TestLinearEncoder:
    """Tests for the LinearEncoder class."""

    def test_dim_mismatch(self, device):
        """Test that encoder raises error for inconsistent input dimensions."""
        pcm_id = 2
        pcm, k, _, _ = load_parity_check_examples(pcm_id)
        bs = 20
        enc = LinearEncoder(pcm, is_pcm=True, device=device)

        # Test for invalid input shape (wrong last dimension)
        with pytest.raises(ValueError, match="Last dimension must be of size"):
            enc(torch.zeros(bs, k + 1, device=device))

    def test_non_binary_matrix_gm(self):
        """Test that encoder raises error for non-binary generator matrix."""
        pcm_id = 2
        pcm, _, _, _ = load_parity_check_examples(pcm_id)
        pcm_modified = np.copy(pcm)
        pcm_modified[0, 0] = 2

        # Test for non-binary matrix (interpreted as gm)
        with pytest.raises(ValueError, match="enc_mat is not binary"):
            LinearEncoder(pcm_modified)

    def test_non_binary_matrix_pcm(self):
        """Test that encoder raises error for non-binary parity-check matrix."""
        pcm_id = 2
        pcm, _, _, _ = load_parity_check_examples(pcm_id)
        pcm_modified = np.copy(pcm)
        pcm_modified[0, 0] = 2

        # Test for non-binary matrix (as pcm)
        with pytest.raises(ValueError, match="enc_mat is not binary"):
            LinearEncoder(pcm_modified, is_pcm=True)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        pcm_id = 2
        pcm, k, _, _ = load_parity_check_examples(pcm_id)
        bs = 20
        enc = LinearEncoder(pcm, is_pcm=True, device=device)
        source = BinarySource(device=device)

        u = source([bs, k])

        # Test with torch.compile
        compiled_enc = torch.compile(enc)
        c = compiled_enc(u)
        assert c.shape == (bs, enc.n)

    @pytest.mark.parametrize("dt", [
        torch.float16, torch.float32, torch.float64, torch.int32, torch.int64,
    ])
    def test_dtypes_flexible(self, device, precision, dt):
        """Test that encoder supports variable dtypes and yields same result."""
        pcm_id = 2
        pcm, k, _, _ = load_parity_check_examples(pcm_id)
        bs = 20
        enc_ref = LinearEncoder(pcm, is_pcm=True, precision="single", device=device)
        source = BinarySource(device=device)

        u = source([bs, k])
        c_ref = enc_ref(u)

        enc = LinearEncoder(pcm, is_pcm=True, precision=precision, device=device)
        u_dt = u.to(dt)
        c = enc(u_dt)

        c_32 = c.to(torch.float32)
        assert torch.equal(c_ref, c_32)

    @pytest.mark.parametrize("shape_prefix", [
        [],
        [10, 20, 30],
        [1, 40],
        [10, 2, 3, 4, 3],
    ])
    def test_multi_dimensional(self, device, shape_prefix):
        """Test against arbitrary input shapes.

        The encoder should only operate on axis=-1.
        """
        pcm_id = 3
        pcm, k, n, _ = load_parity_check_examples(pcm_id)
        s = shape_prefix + [k]
        enc = LinearEncoder(pcm, is_pcm=True, device=device)
        source = BinarySource(device=device)

        u = source(s)
        u_ref = u.reshape(-1, k)

        c = enc(u)
        c_ref = enc(u_ref)

        expected_shape = shape_prefix + [n]
        c_ref = c_ref.reshape(expected_shape)
        assert torch.equal(c, c_ref)

    def test_wrong_last_dim_raises(self, device):
        """Test that wrong last dimension raises ValueError."""
        pcm_id = 3
        pcm, k, _, _ = load_parity_check_examples(pcm_id)
        enc = LinearEncoder(pcm, is_pcm=True, device=device)
        source = BinarySource(device=device)

        # Wrong last dimension
        s = [10, 2, k - 1]
        u = source(s)
        with pytest.raises(ValueError, match="Last dimension must be of size"):
            enc(u)

    def test_random_matrices(self, device):
        """Test against random parity-check matrices.

        Verifies that all codewords fulfill all parity-checks.
        """
        n_trials = 100
        bs = 100
        k = 89
        n = 123
        source = BinarySource(device=device)

        for _ in range(n_trials):
            # Sample a random matrix
            pcm = config.np_rng.uniform(low=0, high=2, size=(n - k, n)).astype(int)

            # Catch internal errors due to non-full rank of pcm (randomly sampled!)
            # In this test we only test that if the encoder initialization
            # succeeds, the resulting encoder object produces valid codewords
            try:
                enc = LinearEncoder(pcm, is_pcm=True, device=device)
            except Exception:
                continue  # ignore this pcm realization

            u = source([bs, k])
            c = enc(u)

            # Verify that all codewords fulfill all parity-checks
            c_np = c.cpu().numpy()
            c_expanded = np.expand_dims(c_np, axis=2)
            pcm_expanded = np.expand_dims(pcm.astype(np.float32), axis=0)
            s = np.matmul(pcm_expanded, c_expanded)
            s = np.mod(s, 2)
            assert np.sum(np.abs(s)) == 0

    def test_properties(self):
        """Test that encoder properties return correct values."""
        pcm_id = 0  # (7,4) Hamming code
        pcm, k, n, coderate = load_parity_check_examples(pcm_id)
        enc = LinearEncoder(pcm, is_pcm=True)

        assert enc.k == k
        assert enc.n == n
        assert enc.coderate == k / n
        assert enc.gm.shape == (k, n)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        # Load (7,4) Hamming code
        pcm, k, n, _ = load_parity_check_examples(0)
        encoder = LinearEncoder(pcm, is_pcm=True)

        # Generate random information bits
        u = torch.randint(0, 2, (10, k), dtype=torch.float32)
        c = encoder(u)
        assert c.shape == torch.Size([10, 7])

    def test_invalid_is_pcm_type(self):
        """Test that non-boolean is_pcm raises TypeError."""
        pcm, _, _, _ = load_parity_check_examples(0)
        with pytest.raises(TypeError, match="is_pcm must be bool"):
            LinearEncoder(pcm, is_pcm="yes")

    def test_invalid_matrix_dimensions(self):
        """Test that 1D matrix raises ValueError."""
        pcm = np.array([1, 0, 1, 1])
        with pytest.raises(ValueError, match="enc_mat must be 2-D array"):
            LinearEncoder(pcm)


