#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for TBDecoder."""

import pytest
import numpy as np
import torch

from sionna.phy.nr import TBEncoder, TBDecoder
from sionna.phy.mapping import BinarySource


class TestTBDecoder:
    """Test TBDecoder."""

    def test_identity(self):
        """Test that receiver can recover info bits."""
        source = BinarySource()

        # Test parameters covering:
        # 1.) Single CB segmentation
        # 2.) Long CB / multiple CWs
        # 3.) Deactivated scrambler
        # 4.) N-dimensional inputs
        # 5.) Zero padding

        bs_list = [[10], [10], [10], [10, 13, 14], [2]]
        tb_sizes = [6656, 60456, 984, 984, 50000]
        num_coded_bits_list = [13440, 100800, 2880, 2880, 100000]
        num_bits_per_symbols = [4, 8, 2, 2, 4]
        num_layers_list = [1, 1, 2, 4, 2]
        n_rntis = [1337, 45678, 1337, 1337, 1337]
        sc_ids = [1, 1023, 2, 42, 42]
        use_scramblers = [True, True, False, True, True]

        for i in range(len(tb_sizes)):
            encoder = TBEncoder(
                target_tb_size=tb_sizes[i],
                num_coded_bits=num_coded_bits_list[i],
                target_coderate=tb_sizes[i] / num_coded_bits_list[i],
                num_bits_per_symbol=num_bits_per_symbols[i],
                num_layers=num_layers_list[i],
                n_rnti=n_rntis[i],
                n_id=sc_ids[i],
                channel_type="PUSCH",
                codeword_index=0,
                use_scrambler=use_scramblers[i],
            )

            decoder = TBDecoder(encoder, num_bp_iter=10, cn_update="minsum")

            u = source(bs_list[i] + [encoder.k])
            c = encoder(u)
            llr_ch = 2 * c - 1  # Apply BPSK
            u_hat, crc_status = decoder(llr_ch)

            # All info bits can be recovered
            np.testing.assert_array_equal(u.cpu().numpy(), u_hat.cpu().numpy())
            # All CRC checks are valid
            assert crc_status.all()

    def test_scrambling(self):
        """Test that (de-)scrambling works as expected."""
        source = BinarySource()
        bs = 10

        n_rnti_ref = 1337
        sc_id_ref = 42

        # Add offset to both scrambling indices
        n_rnti_offset = [0, 1, 0]
        sc_id_offset = [0, 0, 1]

        decoder = None
        for i, _ in enumerate(n_rnti_offset):
            encoder = TBEncoder(
                target_tb_size=60456,
                num_coded_bits=100800,
                target_coderate=60456 / 100800,
                num_bits_per_symbol=4,
                n_rnti=n_rnti_ref + n_rnti_offset[i],
                n_id=sc_id_ref + sc_id_offset[i],
                use_scrambler=True,
            )

            if decoder is None:  # Init decoder only once
                decoder = TBDecoder(encoder, num_bp_iter=20, cn_update="minsum")

            # As scrambling IDs do not match, all TBs must be wrong
            if i > 0:
                u = source([bs, encoder.k])
                c = encoder(u)
                llr_ch = 2 * c - 1  # Apply BPSK
                u_hat, crc_status = decoder(llr_ch)

                # All info bits cannot be recovered
                assert not np.array_equal(u.cpu().numpy(), u_hat.cpu().numpy())
                # All CRC checks are wrong
                assert not crc_status.any()

    def test_crc(self):
        """Test that CRC detects the correct erroneous positions."""
        source = BinarySource()
        bs = 10

        encoder = TBEncoder(
            target_tb_size=60456,
            num_coded_bits=100800,
            target_coderate=60456 / 100800,
            num_bits_per_symbol=4,
            n_rnti=12367,
            n_id=312,
            use_scrambler=True,
        )

        decoder = TBDecoder(encoder, num_bp_iter=20, cn_update="minsum")

        u = source([bs, encoder.k])
        c = encoder(u)
        llr_ch = 2 * c - 1  # Apply BPSK

        # Destroy TB at batch index 7
        # All others are correctly received
        err_pos = 7
        llr_ch_np = llr_ch.cpu().numpy()
        llr_ch_np[err_pos, 500:590] = -10  # Overwrite some LLR positions

        llr_ch_corrupt = torch.tensor(
            llr_ch_np, dtype=llr_ch.dtype, device=llr_ch.device
        )
        u_hat, crc_status = decoder(llr_ch_corrupt)

        # All CRC checks are correct except at err_pos
        crc_status_np = crc_status.cpu().numpy()
        crc_status_ref = np.ones_like(crc_status_np)
        crc_status_ref[err_pos] = 0

        np.testing.assert_array_equal(crc_status_np, crc_status_ref)

    def test_torch_compile(self):
        """Test torch.compile (equivalent to TF's tf.function/XLA)."""
        source = BinarySource()
        bs = 10

        encoder = TBEncoder(
            target_tb_size=60456,
            num_coded_bits=100800,
            target_coderate=60456 / 100800,
            num_bits_per_symbol=4,
            n_rnti=12367,
            n_id=312,
            use_scrambler=True,
        )

        decoder = TBDecoder(encoder, num_bp_iter=20, cn_update="minsum")

        # Test basic functionality
        u = source([bs, encoder.k])
        c = encoder(u)
        llr_ch = 2 * c - 1
        x = decoder(llr_ch)
        assert x[0].shape[0] == bs

        # Change batch_size
        u2 = source([2 * bs, encoder.k])
        c2 = encoder(u2)
        llr_ch2 = 2 * c2 - 1
        x2 = decoder(llr_ch2)
        assert x2[0].shape[0] == 2 * bs

        # Test with torch.compile
        compiled_decoder = torch.compile(decoder)

        u3 = source([bs, encoder.k])
        c3 = encoder(u3)
        llr_ch3 = 2 * c3 - 1
        x3 = compiled_decoder(llr_ch3)
        assert x3[0].shape[0] == bs

        # Change batch_size with compiled
        u4 = source([2 * bs, encoder.k])
        c4 = encoder(u4)
        llr_ch4 = 2 * c4 - 1
        x4 = compiled_decoder(llr_ch4)
        assert x4[0].shape[0] == 2 * bs

