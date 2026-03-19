#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for LayerMapper and LayerDemapper."""

import pytest
import numpy as np
import torch

from sionna.phy.nr import LayerMapper, LayerDemapper
from sionna.phy.utils import hard_decisions
from sionna.phy.mapping import Mapper, Demapper, BinarySource


class TestLayerMapper:
    """Tests for LayerMapper."""

    def test_single_layer(self):
        """Test 1 layer mapping."""
        u = np.array([[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                       0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
                       1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
                       0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                       0, 1, 0, 0, 1, 1]])

        o1 = np.array([[[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
                         0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
                         1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
                         1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
                         1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                         1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
                         0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]]])

        mapper = LayerMapper(num_layers=1)
        u_tensor = torch.tensor(u, dtype=torch.float32)
        o = mapper(u_tensor)
        np.testing.assert_array_equal(o.cpu().numpy(), o1)

    def test_two_layers(self):
        """Test 2 layer mapping."""
        u = np.array([[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                       0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
                       1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
                       0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                       0, 1, 0, 0, 1, 1]])

        o2 = np.array([[[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1,
                         1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
                         1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
                         0, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
                         1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1,
                         1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                         1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]]])

        mapper = LayerMapper(num_layers=2)
        u_tensor = torch.tensor(u, dtype=torch.float32)
        o = mapper(u_tensor)
        np.testing.assert_array_equal(o.cpu().numpy(), o2)

    def test_three_layers(self):
        """Test 3 layer mapping."""
        u = np.array([[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                       0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
                       1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
                       0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                       0, 1, 0, 0, 1, 1]])

        o3 = np.array([[[1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                         1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                         0, 1, 0, 0], [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
                         1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                         1, 1, 0, 0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1, 1,
                         1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                         0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]]])

        mapper = LayerMapper(num_layers=3)
        u_tensor = torch.tensor(u, dtype=torch.float32)
        o = mapper(u_tensor)
        np.testing.assert_array_equal(o.cpu().numpy(), o3)

    def test_four_layers(self):
        """Test 4 layer mapping."""
        u = np.array([[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
                       0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0,
                       0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
                       1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
                       0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                       0, 1, 0, 0, 1, 1]])

        o4 = np.array([[[1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                         0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0, 1,
                         1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0,
                         1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                         0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1], [1, 0,
                         1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                         1, 0, 1, 1, 0, 0, 1, 1, 1]]])

        mapper = LayerMapper(num_layers=4)
        u_tensor = torch.tensor(u, dtype=torch.float32)
        o = mapper(u_tensor)
        np.testing.assert_array_equal(o.cpu().numpy(), o4)

    def test_five_layers_dual_cw(self):
        """Test 5 layer mapping (dual codeword)."""
        u1 = np.array([[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                        0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 1, 1, 1]])

        u2 = np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
                        1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                        1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                        1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                        0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        o5 = np.array([[[0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
                         0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0,
                         1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
                         0, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1, 0, 0, 0,
                         1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0,
                         0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                         0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1,
                         1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
                         1, 1, 1, 1, 1, 0, 1, 0, 1, 0]]])

        mapper = LayerMapper(num_layers=5)
        u1_tensor = torch.tensor(u1, dtype=torch.float32)
        u2_tensor = torch.tensor(u2, dtype=torch.float32)
        o = mapper([u1_tensor, u2_tensor])
        np.testing.assert_array_equal(o.cpu().numpy(), o5)

    def test_six_layers_dual_cw(self):
        """Test 6 layer mapping (dual codeword)."""
        u1 = np.array([[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                        0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1,
                        1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
                        0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]])

        u2 = np.array([[0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
                        0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0,
                        1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                        1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0]])

        o6 = np.array([[[1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
                         1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 0,
                         1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0,
                         0, 1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                         0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0,
                         1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1,
                         1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
                         1, 0, 0, 0, 0, 0, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0, 0,
                         1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                         1, 0, 0, 1, 0]]])

        mapper = LayerMapper(num_layers=6)
        u1_tensor = torch.tensor(u1, dtype=torch.float32)
        u2_tensor = torch.tensor(u2, dtype=torch.float32)
        o = mapper([u1_tensor, u2_tensor])
        np.testing.assert_array_equal(o.cpu().numpy(), o6)

    def test_seven_layers_dual_cw(self):
        """Test 7 layer mapping (dual codeword)."""
        u1 = np.array([[1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
                        1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                        0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                        0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                        1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]])

        u2 = np.array([[0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
                        1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
                        0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                        1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                        1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                        0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]])

        o7 = np.array([[[1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1,
                      0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0,
                      1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                      0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                      1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1], [0,
                      1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0,
                      1, 1, 1, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1,
                      1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                      0, 1, 0], [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
                      1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 1,
                      1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1, 1, 0]]])

        mapper = LayerMapper(num_layers=7)
        u1_tensor = torch.tensor(u1, dtype=torch.float32)
        u2_tensor = torch.tensor(u2, dtype=torch.float32)
        o = mapper([u1_tensor, u2_tensor])
        np.testing.assert_array_equal(o.cpu().numpy(), o7)

    def test_eight_layers_dual_cw(self):
        """Test 8 layer mapping (dual codeword)."""
        u1 = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
                        0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,
                        1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                        0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
                        1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
                        0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0,
                        1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]])

        u2 = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,
                        0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
                        0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,
                        1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                        1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
                        1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]])

        o8 = np.array([[[0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                      1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0,
                      1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1,
                      1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                      0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0], [1,
                      0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
                      1, 0, 1, 1, 1, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 0, 1, 0,
                      1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                      1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
                      0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], [0, 0, 0,
                      0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1, 1, 1, 0,
                      1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1,
                      1]]])

        mapper = LayerMapper(num_layers=8)
        u1_tensor = torch.tensor(u1, dtype=torch.float32)
        u2_tensor = torch.tensor(u2, dtype=torch.float32)
        o = mapper([u1_tensor, u2_tensor])
        np.testing.assert_array_equal(o.cpu().numpy(), o8)


class TestLayerDemapper:
    """Tests for LayerDemapper."""

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
    def test_identity_single_cw(self, num_layers):
        """Test that original sequence can be recovered with single codeword."""
        bs = 7
        source = BinarySource()
        mapper = LayerMapper(num_layers=num_layers)
        demapper = LayerDemapper(mapper)

        n = 12 * num_layers  # Divisible by all layer counts
        x = source([bs, n])
        y = mapper(x)
        z = demapper(y)

        np.testing.assert_array_equal(x.cpu().numpy(), z.cpu().numpy())

    @pytest.mark.parametrize("num_layers,n0,n1", [
        (5, 60, 90),
        (6, 90, 90),
        (7, 90, 120),
        (8, 120, 120),
    ])
    def test_identity_dual_cw(self, num_layers, n0, n1):
        """Test recovery with dual codeword configurations."""
        bs = 7
        source = BinarySource()
        mapper = LayerMapper(num_layers=num_layers)
        demapper = LayerDemapper(mapper)

        x0 = source([bs, n0])
        x1 = source([bs, n1])

        y = mapper([x0, x1])
        z0, z1 = demapper(y)

        np.testing.assert_array_equal(x0.cpu().numpy(), z0.cpu().numpy())
        np.testing.assert_array_equal(x1.cpu().numpy(), z1.cpu().numpy())

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
    def test_higher_order_modulation(self, num_layers):
        """Test LLRs are correctly grouped with higher-order modulation."""
        bs = 10
        mod_order = 4

        source = BinarySource()
        mapper = Mapper("qam", num_bits_per_symbol=mod_order)
        demapper = Demapper("maxlog", "qam", num_bits_per_symbol=mod_order)

        l_mapper = LayerMapper(num_layers=num_layers)
        l_demapper = LayerDemapper(l_mapper, num_bits_per_symbol=mod_order)

        u = source((bs, 5, 7, 12 * num_layers * mod_order))
        x = mapper(u)
        x_l = l_mapper(x)
        no = torch.tensor(0.1, device=x_l.device)
        llr_l = demapper(x_l, no)
        l_hat = l_demapper(llr_l)
        u_hat = hard_decisions(l_hat)

        np.testing.assert_array_equal(u.cpu().numpy(), u_hat.cpu().numpy())

    @pytest.mark.parametrize("num_layers,n0,n1", [
        (5, 60, 90),
        (6, 90, 90),
        (7, 90, 120),
        (8, 120, 120),
    ])
    def test_higher_order_modulation_dual_cw(self, num_layers, n0, n1):
        """Test higher-order modulation with dual codeword configurations."""
        bs = 10
        mod_order = 4

        source = BinarySource()
        mapper = Mapper("qam", num_bits_per_symbol=mod_order)
        demapper = Demapper("maxlog", "qam", num_bits_per_symbol=mod_order)

        l_mapper = LayerMapper(num_layers=num_layers)
        l_demapper = LayerDemapper(l_mapper, num_bits_per_symbol=mod_order)

        u0 = source([bs, n0 * mod_order * num_layers])
        u1 = source([bs, n1 * mod_order * num_layers])
        x0 = mapper(u0)
        x1 = mapper(u1)

        y = l_mapper([x0, x1])
        no = torch.tensor(0.01, device=y.device)
        llr = demapper(y, no)
        z0, z1 = l_demapper(llr)

        u_hat0 = hard_decisions(z0)
        u_hat1 = hard_decisions(z1)

        np.testing.assert_array_equal(u0.cpu().numpy(), u_hat0.cpu().numpy())
        np.testing.assert_array_equal(u1.cpu().numpy(), u_hat1.cpu().numpy())

