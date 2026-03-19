#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.utils"""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.sys.utils import get_pathloss, is_scheduled_in_slot, spread_across_subcarriers


class TestSysUtils:
    """Tests for sys utility functions."""

    def test_is_scheduled_in_slot_sinr(self, device, precision):
        """Test is_scheduled_in_slot with SINR input."""
        batch_size = 2
        num_ofdm_symbols = 14
        num_subcarriers = 52
        num_ut = 4
        num_streams_per_ut = 2

        # Create SINR tensor with some zeros (unscheduled)
        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        )
        # Set user 1 to unscheduled (all zeros)
        sinr[:, :, :, 1, :] = 0

        is_sched = is_scheduled_in_slot(sinr=sinr)

        # Check shape
        assert is_sched.shape == (batch_size, num_ut)

        # User 1 should not be scheduled
        assert not is_sched[:, 1].any()

        # Other users should be scheduled
        assert is_sched[:, 0].all()
        assert is_sched[:, 2].all()
        assert is_sched[:, 3].all()

    def test_is_scheduled_in_slot_num_re(self, device):
        """Test is_scheduled_in_slot with num_allocated_re input."""
        batch_size = 2
        num_ut = 4

        num_allocated_re = torch.tensor(
            [[10, 0, 5, 8], [0, 12, 3, 0]], dtype=torch.int32, device=device
        )

        is_sched = is_scheduled_in_slot(num_allocated_re=num_allocated_re)

        expected = torch.tensor(
            [[True, False, True, True], [False, True, True, False]], device=device
        )
        assert (is_sched == expected).all()

    def test_get_pathloss(self, device, precision):
        """Test get_pathloss function."""
        batch_size = 2
        num_rx = 4
        num_rx_ant = 1
        num_tx = 4
        num_tx_ant = 4
        num_ofdm_sym = 14
        num_subcarriers = 52

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_sym,
            num_subcarriers,
            dtype=cdtype,
            device=device,
        )

        pathloss_all, pathloss_serving = get_pathloss(h_freq, precision=precision)

        # Check shape
        assert pathloss_all.shape == (batch_size, num_rx, num_tx, num_ofdm_sym)

        # Check that pathloss is positive
        assert (pathloss_all > 0).all()

    def test_get_pathloss_with_association(self, device, precision):
        """Test get_pathloss with RX-TX association matrix."""
        batch_size = 2
        num_rx = 4
        num_rx_ant = 1
        num_tx = 4
        num_tx_ant = 4
        num_ofdm_sym = 14
        num_subcarriers = 52

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        h_freq = torch.randn(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_ant,
            num_ofdm_sym,
            num_subcarriers,
            dtype=cdtype,
            device=device,
        )

        # Diagonal association (each RX connected to one TX)
        rx_tx_association = torch.eye(num_rx, num_tx, dtype=torch.int32, device=device)

        pathloss_all, pathloss_serving = get_pathloss(
            h_freq, rx_tx_association=rx_tx_association, precision=precision
        )

        # Check shapes
        assert pathloss_all.shape == (batch_size, num_rx, num_tx, num_ofdm_sym)
        assert pathloss_serving.shape == (batch_size, num_rx, num_ofdm_sym)

    def test_spread_across_subcarriers(self, device, precision):
        """Test spread_across_subcarriers function."""
        from sionna.phy import dtypes
        dtype = dtypes[precision]["torch"]["dtype"]
        
        batch_size = 2
        num_ofdm_sym = 14
        num_ut = 4
        num_subcarriers = 52
        num_streams = 2

        tx_power_per_ut = torch.ones(
            batch_size, num_ofdm_sym, num_ut, dtype=dtype, device=device
        )
        is_scheduled = torch.ones(
            batch_size,
            num_ofdm_sym,
            num_subcarriers,
            num_ut,
            num_streams,
            dtype=torch.bool,
            device=device,
        )

        tx_power = spread_across_subcarriers(
            tx_power_per_ut, is_scheduled, precision=precision
        )

        # Check shape
        assert tx_power.shape == (batch_size, num_ut, num_streams, num_ofdm_sym, num_subcarriers)

        # Total power per user should equal input power
        total_power = tx_power.sum(dim=(-1, -2, -3))
        expected = tx_power_per_ut.sum(dim=1)
        assert torch.allclose(total_power, expected, rtol=1e-4)

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_is_scheduled_compiled(self, device, mode):
        """Test that is_scheduled_in_slot works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        batch_size = 2
        num_ofdm_symbols = 4
        num_subcarriers = 12
        num_ut = 4
        num_streams_per_ut = 2

        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        )

        # Compile the function
        if mode != "default":
            compiled_fn = torch.compile(is_scheduled_in_slot, mode=mode)
        else:
            compiled_fn = is_scheduled_in_slot

        is_sched = compiled_fn(sinr=sinr)

        assert is_sched.shape == (batch_size, num_ut)
