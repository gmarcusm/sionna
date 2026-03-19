#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.polar.utils."""

import numpy as np
import pytest
import torch

from sionna.phy.fec.polar.utils import (
    generate_5g_ranking,
    generate_polar_transform_mat,
    generate_rm_code,
    generate_dense_polar,
)
from sionna.phy.fec.polar import PolarEncoder
from sionna.phy.mapping import BinarySource


class TestGenerate5GRanking:
    """Tests for generate_5g_ranking function."""

    @pytest.mark.parametrize(
        "k,n",
        [
            (-1, 32),
            (10, -3),
            (1.0, 32),
            (3, 32.0),
            (33, 32),
            (10, 31),
            (1025, 2048),
            (16, 33),
            (7, 16),
            (1000, 2048),
        ],
    )
    def test_invalid_inputs(self, k, n):
        """Test against invalid values of n and k."""
        with pytest.raises(BaseException):
            generate_5g_ranking(k, n)

    @pytest.mark.parametrize(
        "k,n",
        [(1, 512), (10, 32), (1000, 1024), (3, 256), (10, 64), (0, 32), (1024, 1024)],
    )
    def test_valid_inputs(self, k, n):
        """Test that valid parameters work."""
        frozen_pos, info_pos = generate_5g_ranking(k, n)
        assert len(frozen_pos) == n - k
        assert len(info_pos) == k

    def test_sort_option(self):
        """Test that sort option returns sorted indices."""
        k, n = 64, 128
        frozen_pos_sorted, info_pos_sorted = generate_5g_ranking(k, n, sort=True)
        frozen_pos_unsorted, info_pos_unsorted = generate_5g_ranking(
            k, n, sort=False
        )

        # Sorted version should be in ascending order
        assert np.array_equal(frozen_pos_sorted, np.sort(frozen_pos_sorted))
        assert np.array_equal(info_pos_sorted, np.sort(info_pos_sorted))

        # Both should contain the same positions
        assert set(frozen_pos_sorted) == set(frozen_pos_unsorted)
        assert set(info_pos_sorted) == set(info_pos_unsorted)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        frozen_pos, info_pos = generate_5g_ranking(k=100, n=256)
        assert len(frozen_pos) == 156
        assert len(info_pos) == 100


class TestGeneratePolarTransformMat:
    """Tests for generate_polar_transform_mat function."""

    def test_invalid_inputs(self):
        """Test against invalid inputs."""
        with pytest.raises(ValueError):
            generate_polar_transform_mat(-1)
        with pytest.raises(ValueError):
            generate_polar_transform_mat(20)  # Too large
        with pytest.raises(ValueError):
            generate_polar_transform_mat(1.5)  # Not integer

    @pytest.mark.parametrize("n_lift", range(1, 10))
    def test_valid_outputs(self, n_lift):
        """Test that valid outputs are generated."""
        gm = generate_polar_transform_mat(n_lift)
        expected_size = 2**n_lift
        assert gm.shape == (expected_size, expected_size)
        assert np.all((gm == 0) | (gm == 1))

    def test_base_case(self):
        """Test the base kernel [[1, 0], [1, 1]]."""
        gm = generate_polar_transform_mat(1)
        expected = np.array([[1, 0], [1, 1]])
        assert np.array_equal(gm, expected)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        gm = generate_polar_transform_mat(3)
        assert gm.shape == (8, 8)


class TestGenerateRMCode:
    """Tests for generate_rm_code function."""

    def test_invalid_inputs(self):
        """Test against invalid inputs."""
        with pytest.raises(TypeError):
            generate_rm_code(1.5, 3)  # r not int
        with pytest.raises(TypeError):
            generate_rm_code(1, 3.5)  # m not int
        with pytest.raises(ValueError):
            generate_rm_code(4, 3)  # r > m
        with pytest.raises(ValueError):
            generate_rm_code(-1, 3)  # r < 0
        with pytest.raises(ValueError):
            generate_rm_code(1, -1)  # m < 0

    @pytest.mark.parametrize(
        "r,m,expected_n,expected_k,expected_d_min",
        [
            (0, 0, 1, 1, 1),
            (1, 1, 2, 2, 1),
            (2, 2, 4, 4, 1),
            (3, 3, 8, 8, 1),
            (4, 4, 16, 16, 1),
            (5, 5, 32, 32, 1),
            (0, 1, 2, 1, 2),
            (1, 2, 4, 3, 2),
            (2, 3, 8, 7, 2),
            (3, 4, 16, 15, 2),
            (4, 5, 32, 31, 2),
            (0, 2, 4, 1, 4),
            (1, 3, 8, 4, 4),
            (2, 4, 16, 11, 4),
            (3, 5, 32, 26, 4),
            (0, 3, 8, 1, 8),
            (1, 4, 16, 5, 8),
            (2, 5, 32, 16, 8),
            (0, 4, 16, 1, 16),
            (1, 5, 32, 6, 16),
            (0, 5, 32, 1, 32),
        ],
    )
    def test_known_parameters(self, r, m, expected_n, expected_k, expected_d_min):
        """Test that Reed-Muller code design yields valid constructions.

        We test against the parameters from
        https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code
        """
        frozen_pos, info_pos, n, k, d_min = generate_rm_code(r, m)

        assert n == expected_n
        assert k == expected_k
        assert d_min == expected_d_min
        assert len(frozen_pos) == n - k
        assert len(info_pos) == k

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        frozen_pos, info_pos, n, k, d_min = generate_rm_code(r=1, m=4)
        assert n == 16
        assert k == 5
        assert d_min == 8


class TestGenerateDensePolar:
    """Tests for generate_dense_polar function."""

    def test_invalid_inputs(self):
        """Test against invalid inputs."""
        frozen_pos = np.array([0, 1, 2], dtype=int)
        # n not a number
        with pytest.raises(TypeError):
            generate_dense_polar(frozen_pos, "8")
        # frozen_pos not int
        with pytest.raises(TypeError):
            generate_dense_polar(np.array([0.5, 1.0]), 8)
        # frozen_pos too long
        with pytest.raises(ValueError):
            generate_dense_polar(np.arange(10), 8)
        # n not a power of 2
        with pytest.raises(ValueError):
            generate_dense_polar(frozen_pos, 10)

    @pytest.mark.parametrize("n", [32, 64, 128, 256])
    @pytest.mark.parametrize("rate", [0.1, 0.5, 0.9])
    def test_valid_outputs(self, device, n, rate):
        """Test naive (dense) polar code construction method."""
        batch_size = 100
        source = BinarySource(device=device)
        k = int(n * rate)

        frozen_pos, _ = generate_5g_ranking(k, n)

        pcm, gm = generate_dense_polar(frozen_pos, n, verbose=False)

        gm_t = torch.tensor(gm, dtype=torch.float32, device=device)
        pcm_t = torch.tensor(pcm, dtype=torch.float32, device=device)

        assert pcm.shape[0] == n - k
        assert pcm.shape[1] == n
        assert gm.shape[0] == k
        assert gm.shape[1] == n

        s = np.mod(np.matmul(pcm, np.transpose(gm)), 2)
        assert np.sum(s) == 0

        encoder = PolarEncoder(frozen_pos, n, device=device)

        u = source([batch_size, k])
        c = encoder(u)

        c_new = torch.matmul(u.unsqueeze(1).float(), gm_t)
        c_new = torch.fmod(c_new.squeeze(1), 2)

        assert torch.equal(c.float(), c_new)

        s = torch.matmul(pcm_t, c.float().T)
        s = torch.fmod(s, 2)
        assert torch.equal(s, torch.zeros_like(s))

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        frozen_pos, _ = generate_5g_ranking(k=32, n=64)
        pcm, gm = generate_dense_polar(frozen_pos, n=64, verbose=False)
        assert pcm.shape == (32, 64)
        assert gm.shape == (32, 64)



