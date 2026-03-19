#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.nr.utils functions."""

import pytest
import numpy as np
import torch

from sionna.phy.nr.utils import decode_mcs_index, generate_prng_seq, calculate_tb_size
from .utils import calculate_tb_size_numpy, decode_mcs_index_numpy


class TestGeneratePrngSeq:
    """Tests for the pseudo-random sequence generator."""

    def test_invalid_length_negative(self):
        """Test rejection of negative length."""
        with pytest.raises(ValueError):
            generate_prng_seq(-1, 10)

    def test_invalid_c_init_negative(self):
        """Test rejection of negative c_init."""
        with pytest.raises(ValueError):
            generate_prng_seq(10, -1)

    def test_invalid_c_init_too_large(self):
        """Test rejection of c_init >= 2^32."""
        with pytest.raises(ValueError):
            generate_prng_seq(100, 2**32)

    def test_reference_sequence(self):
        """Test against reference example from 3GPP."""
        n_rnti = 20001
        n_id = 41
        c_init = n_rnti * 2**15 + n_id
        length = 100

        s_ref = np.array([
            0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
            1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
            1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.,
            0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
            0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0.,
            1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
            1., 1., 1., 1., 1., 1., 1., 0., 0.
        ])

        s = generate_prng_seq(length, c_init)
        np.testing.assert_array_equal(s, s_ref)

    def test_different_c_init_different_sequence(self):
        """Test that different c_init produces different sequence."""
        n_rnti = 20001
        n_id = 41
        c_init = n_rnti * 2**15 + n_id
        length = 100

        s1 = generate_prng_seq(length, c_init)
        s2 = generate_prng_seq(length, c_init + 1)

        assert not np.array_equal(s1, s2)


class TestDecodeMcsIndex:
    """Tests for MCS index decoding."""

    def test_pdsch_table1(self):
        """Test PDSCH MCS table 1 (Table 5.1.3.1-1)."""
        qs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 438, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx, table_index=1, is_pusch=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pdsch_table2(self):
        """Test PDSCH MCS table 2 (Table 5.1.3.1-2)."""
        qs = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6,
              6, 6, 8, 8, 8, 8, 8, 8, 8, 8]
        rs = [120, 193, 308, 449, 602, 378, 434, 490, 553, 616,
              658, 466, 517, 567, 616, 666, 719, 772, 822, 873,
              682.5, 711, 754, 797, 841, 885, 916.5, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx, table_index=2, is_pusch=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pusch_without_precoding_table1(self):
        """Test PUSCH without transform precoding (Table 5.1.3.1-1)."""
        qs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 438, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(
                mcs_index=idx, table_index=1, is_pusch=True,
                transform_precoding=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pusch_with_precoding_table1_pi2bpsk_false(self):
        """Test PUSCH with transform precoding Table 6.1.4.1-1 (q=2)."""
        qs = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(
                mcs_index=idx, table_index=1, is_pusch=True,
                transform_precoding=True, pi2bpsk=False)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_pusch_with_precoding_table1_pi2bpsk_true(self):
        """Test PUSCH with transform precoding Table 6.1.4.1-1 (q=1)."""
        qs = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
              6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        rs = [240, 314, 193, 251, 308, 379, 449, 526, 602, 679,
              340, 378, 434, 490, 553, 616, 658, 466, 517,
              567, 616, 666, 719, 772, 822, 873, 910, 948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(
                mcs_index=idx, table_index=1, is_pusch=True,
                transform_precoding=True, pi2bpsk=True)
            assert m.item() == q
            assert r.item() == pytest.approx(rs[idx] / 1024)

    def test_invalid_mcs_index_raises(self):
        """Test that invalid MCS index raises error."""
        with pytest.raises(AssertionError):
            decode_mcs_index(mcs_index=29, table_index=1)


class TestCalculateTbSize:
    """Tests for transport block size calculation."""

    @pytest.mark.parametrize("mcs_index", [0, 4, 16, 20, 27])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("num_prbs", [1, 20, 100])
    def test_tb_size_consistency(self, mcs_index, num_layers, num_prbs):
        """Test TB size calculation produces consistent results."""
        q, r = decode_mcs_index(mcs_index, table_index=2)
        # Convert tensors to Python scalars
        q_val = q.item()
        r_val = r.item()
        num_ofdm_symbols = 14
        num_dmrs_per_prb = 12

        result = calculate_tb_size(
            target_coderate=r_val,
            modulation_order=q_val,
            num_layers=num_layers,
            num_prbs=num_prbs,
            num_ofdm_symbols=num_ofdm_symbols,
            num_dmrs_per_prb=num_dmrs_per_prb,
            verbose=False,
        )

        tb_size, cb_size, num_cbs, tb_crc_length, cb_crc_length, cw_length = result
        # Convert tensor results to Python scalars
        tb_size = int(tb_size)
        cb_size = int(cb_size)
        num_cbs = int(num_cbs)
        tb_crc_length = int(tb_crc_length)
        cb_crc_length = int(cb_crc_length)

        # TB size must equal number of CB bits (+CRC overhead)
        assert tb_size == num_cbs * (cb_size - cb_crc_length) - tb_crc_length

        # Individual cw length for each cb
        assert num_cbs == len(cw_length)

        # Single cw TB has no CB CRC
        if num_cbs == 1:
            assert cb_crc_length == 0
        else:
            assert cb_crc_length == 24

        # TB CRC is 16 or 24
        if tb_size > 3824:
            assert tb_crc_length == 24
        else:
            assert tb_crc_length == 16

    def test_tb_size_vs_numpy(self):
        """Validate calculate_tb_size against NumPy reference."""
        q, r = decode_mcs_index(10, table_index=1)
        # Convert tensors to Python scalars
        q_val = q.item()
        r_val = r.item()

        result = calculate_tb_size(
            target_coderate=r_val,
            modulation_order=q_val,
            num_layers=1,
            num_prbs=50,
            num_ofdm_symbols=14,
            num_dmrs_per_prb=12,
            verbose=False,
        )
        tb_size, cb_size, num_cbs, tb_crc_length, cb_crc_length, cw_length = result
        # Convert tensor results to Python scalars
        tb_size = int(tb_size)
        cb_size = int(cb_size)
        num_cbs = int(num_cbs)
        tb_crc_length = int(tb_crc_length)
        cb_crc_length = int(cb_crc_length)
        if isinstance(cw_length, torch.Tensor):
            cw_length = cw_length.cpu().numpy()

        # Compare against numpy version
        result_np = calculate_tb_size_numpy(
            modulation_order=q_val,
            target_coderate=r_val,
            num_layers=1,
            num_prbs=50,
            num_ofdm_symbols=14,
            num_dmrs_per_prb=12,
            verbose=False,
        )

        assert tb_size == result_np[0]
        assert cb_size == result_np[1]
        assert num_cbs == result_np[2]
        assert tb_crc_length == result_np[3]
        assert cb_crc_length == result_np[4]
        np.testing.assert_array_equal(cw_length, result_np[5])


class TestDecodeMcsIndexAgainstNumpy:
    """Test decode_mcs_index against NumPy reference implementation."""

    @pytest.mark.parametrize("mcs_index", range(0, 27))
    @pytest.mark.parametrize("table_index", [1, 2])
    @pytest.mark.parametrize("is_pusch", [True, False])
    def test_vs_numpy(self, mcs_index, table_index, is_pusch):
        """Compare PyTorch version against NumPy reference."""
        # PyTorch version
        m_torch, r_torch = decode_mcs_index(
            mcs_index=mcs_index,
            table_index=table_index,
            is_pusch=is_pusch,
            transform_precoding=False,
        )

        # NumPy reference
        channel_type = "PUSCH" if is_pusch else "PDSCH"
        m_np, r_np = decode_mcs_index_numpy(
            mcs_index=mcs_index,
            table_index=table_index,
            channel_type=channel_type,
            transform_precoding=False,
        )

        assert m_torch.item() == m_np
        assert r_torch.item() == pytest.approx(r_np)

