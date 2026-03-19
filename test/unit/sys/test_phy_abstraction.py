#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.phy_abstraction"""

import numpy as np
import os
import pytest
import torch

from sionna.phy import config
from sionna.phy.utils import DeepUpdateDict, random_tensor_from_values
from sionna.sys import PHYAbstraction


class TestPHYAbstraction:
    """Tests for the PHYAbstraction class."""

    def test_write_and_load(self, device):
        """Test the SNR to BER/BLER table generation."""
        sim_set_1 = {
            "category": {
                0: {"index": {1: {"MCS": [10, 24]}, 2: {"MCS": [12]}}}
            }
        }
        snr_dbs_1 = [0, 20]
        cb_sizes_1 = [50, 100, 150]
        filename = "test.json"

        # Start from no loaded table
        with pytest.warns(UserWarning):
            phy_abs = PHYAbstraction(load_bler_tables_from="", device=device)

        # Compute tables and save them to file
        table_1 = phy_abs.new_bler_table(
            snr_dbs_1,
            cb_sizes_1,
            sim_set_1,
            filename=filename,
            max_mc_iter=15,
            batch_size=10,
            verbose=False,
        )

        # Check that results have been written to file
        assert os.path.isfile(filename), "File was not created"
        assert os.path.getsize(filename) > 0, "File is empty"

        # Load tables
        table_loaded = PHYAbstraction.load_table(filename)

        # Check that the two tables (dumped and loaded) are equal
        for category in table_1["category"]:
            for table_index in table_1["category"][category]["index"]:
                for mcs in table_1["category"][category]["index"][table_index]["MCS"]:
                    res_mcs = table_1["category"][category]["index"][table_index][
                        "MCS"
                    ][mcs]
                    res_mcs1 = table_loaded["category"][category]["index"][
                        table_index
                    ]["MCS"][mcs]

                    for a, b in zip(res_mcs["SNR_db"], res_mcs1["SNR_db"]):
                        assert a == b, "SNR_db mismatch"

                    for cbs in res_mcs["CBS"]:
                        res = res_mcs["CBS"][cbs]
                        res1 = res_mcs1["CBS"][cbs]

                        for a, b in zip(res["BLER"], res1["BLER"]):
                            assert a == b, "BLER mismatch"

        # Remove the file
        if os.path.isfile(filename):
            os.remove(filename)

    def test_bler_interpolation(self, device):
        """Validate the (CBS, SNR) -> BLER interpolation."""
        categories = [1, 1, 1]
        table_index = [1, 1, 1]
        mcs = [10, 15, 16]

        # Instantiate the PHY abstraction object
        phy_abs = PHYAbstraction(device=device)

        assert len(categories) == len(table_index)
        assert len(table_index) == len(mcs)

        for k in range(len(categories)):
            table_tmp = phy_abs.bler_table["category"][categories[k]]["index"][
                table_index[k]
            ]["MCS"][mcs[k]]

            # SNR/CBS values at which tables have been simulated
            snr_dbs_sim = table_tmp["SNR_db"]
            cb_sizes_sim = list(table_tmp["CBS"].keys())

            # Redefine the interpolation grid
            phy_abs.cbs_interp_min_max_delta = (
                cb_sizes_sim[0],
                cb_sizes_sim[-1],
                (cb_sizes_sim[1] - cb_sizes_sim[0]) // 10,
            )
            phy_abs._snr_db_interp_min_max_delta = (
                snr_dbs_sim[0],
                snr_dbs_sim[-1],
                (snr_dbs_sim[1] - snr_dbs_sim[0]) / 10,
            )

            # Interpolated table
            table_interp = phy_abs.bler_table_interp.cpu().numpy()[
                categories[k], table_index[k] - 1, mcs[k], ::
            ]

            for cbs in cb_sizes_sim:
                cbs_interp_ind = np.argmin(abs(phy_abs._cbs_interp - cbs))
                bler_sim = table_tmp["CBS"][cbs]["BLER"]
                for ii, snr in enumerate(snr_dbs_sim):
                    snr_interp_ind = np.argmin(abs(phy_abs._snr_dbs_interp - snr))

                    bler_interp = table_interp[cbs_interp_ind, snr_interp_ind]

                    # Check that interpolated value and original value coincide
                    assert abs(bler_sim[ii] - bler_interp) < 1e-2, (
                        f"BLER mismatch at CBS={cbs}, SNR={snr}"
                    )

    def test_get_bler(self, device):
        """Test get_bler method."""
        cbs_delta = 99
        assert (cbs_delta % 2) != 0, "cbs_delta must be odd"

        phy_abs = PHYAbstraction(
            cbs_interp_min_max_delta=(24, 8448, cbs_delta),
            precision="double",
            device=device,
        )

        # Check that it does not throw errors with non-tensor inputs
        bler_float = phy_abs.get_bler(
            mcs_index=10,
            mcs_table_index=1,
            mcs_category=0,  # PUSCH
            cb_size=500,
            snr_eff=10,
        )

        # Test with tensor inputs
        shape = [20, 20]
        generator = config.torch_rng(device)

        snr_db = torch.rand(shape, device=device, generator=generator) * 20
        snr = torch.pow(torch.tensor(10.0, device=device), snr_db / 10)
        table_index = random_tensor_from_values([1, 2], shape)
        mcs = random_tensor_from_values(list(range(10, 20)), shape)
        cbs = torch.randint(24, 8000, shape, dtype=torch.int32, device=device, generator=generator)
        category = random_tensor_from_values([0, 1], shape)

        bler_pt = phy_abs.get_bler(mcs, table_index, category, cbs, snr)
        bler_pt = bler_pt.cpu().numpy()

        for i1 in range(shape[0]):
            for i2 in range(shape[1]):
                table_idx = table_index[i1, i2].item() - 1
                category_idx = category[i1, i2].item()
                mcs_idx = mcs[i1, i2].item()

                cbs_ = cbs[i1, i2].item()
                cbs_idx = np.argmin(abs(phy_abs._cbs_interp - cbs_))

                snr_db_ = snr_db[i1, i2].item()
                snr_db_idx = np.argmin(abs(phy_abs._snr_dbs_interp - snr_db_))

                bler_numpy = phy_abs.bler_table_interp[
                    category_idx, table_idx, mcs_idx, cbs_idx, snr_db_idx
                ].cpu().numpy()

                assert abs(bler_pt[i1, i2] - bler_numpy) < 1e-6, (
                    f"BLER mismatch at ({i1}, {i2})"
                )

    def test_call(self, device):
        """Ensure that 'call' method of PHYAbstraction does not throw any error."""
        batch_size = 2
        num_ut = 8
        num_ofdm_symbols = 4
        num_subcarriers = 12
        num_streams_per_ut = 2

        # Generate SINR
        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        ) * 100

        # MCS
        mcs_index = torch.randint(
            3, 10, (batch_size, num_ut), dtype=torch.int32, device=device
        )

        mcs_table_index = 1
        mcs_category = 0  # PUSCH

        # Instantiate PHYAbstraction object
        phy_abs = PHYAbstraction(precision="double", device=device)

        num_decoded_bits, harq_feedback, sinr_eff, tbler, bler = phy_abs(
            mcs_index,
            sinr=sinr,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
        )

        # Basic shape checks
        assert num_decoded_bits.shape == (batch_size, num_ut)
        assert harq_feedback.shape == (batch_size, num_ut)
        assert sinr_eff.shape == (batch_size, num_ut)
        assert tbler.shape == (batch_size, num_ut)
        assert bler.shape == (batch_size, num_ut)

        # If HARQ=1 (ACK) then number of successfully decoded bits must be positive
        ack_mask = harq_feedback == 1
        if ack_mask.any():
            assert (num_decoded_bits[ack_mask] > 0).all(), (
                "ACK with zero decoded bits"
            )

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compiled(self, device, mode):
        """Test that PHYAbstraction works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        batch_size = 2
        num_ut = 4
        num_ofdm_symbols = 2
        num_subcarriers = 12
        num_streams_per_ut = 2

        sinr = torch.rand(
            batch_size,
            num_ofdm_symbols,
            num_subcarriers,
            num_ut,
            num_streams_per_ut,
            device=device,
        ) * 100

        mcs_index = torch.randint(
            3, 10, (batch_size, num_ut), dtype=torch.int32, device=device
        )

        phy_abs = PHYAbstraction(device=device)

        # Compile the call method
        if mode != "default":
            compiled_call = torch.compile(phy_abs.call, mode=mode)
        else:
            compiled_call = phy_abs.call

        # Run compiled version
        num_decoded_bits, harq_feedback, sinr_eff, tbler, bler = compiled_call(
            mcs_index, sinr=sinr, mcs_table_index=1, mcs_category=0
        )

        # Basic shape checks
        assert num_decoded_bits.shape == (batch_size, num_ut)
