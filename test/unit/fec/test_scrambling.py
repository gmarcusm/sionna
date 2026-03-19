#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.scrambling module."""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fec.scrambling import Descrambler, Scrambler, TB5GScrambler
from sionna.phy.nr.utils import generate_prng_seq


class TestScrambler:
    """Tests for the Scrambler class."""

    @pytest.mark.parametrize("seq_length", [1, 100, 256, 10000])
    @pytest.mark.parametrize("batch_size", [1, 100, 256])
    def test_sequence_dimension_keep_state_true(self, device, seq_length, batch_size):
        """Test against correct dimensions of the sequence with keep_state=True."""
        s = Scrambler(binary=False, keep_state=True, device=device)
        llr = torch.rand(batch_size, seq_length, device=device)
        x = s(llr)
        assert list(x.shape) == [batch_size, seq_length]

    @pytest.mark.parametrize("seq_length", [1, 100, 256])
    @pytest.mark.parametrize("batch_size", [1, 100])
    def test_sequence_dimension_keep_state_false(self, device, seq_length, batch_size):
        """Test against correct dimensions of the sequence with keep_state=False."""
        s = Scrambler(binary=False, keep_state=False, device=device)
        llr = torch.rand(batch_size, seq_length, device=device)
        x = s(llr)
        assert list(x.shape) == [batch_size, seq_length]

    def test_sequence_dimension_no_batch(self, device):
        """Test non-batch dimension."""
        s = Scrambler(binary=False, keep_state=False, device=device)
        llr = torch.zeros(100, device=device)
        x = s(llr)
        assert x.shape == (100,)

    @pytest.mark.parametrize("seed", [None, 1337, 1234, 1003])
    @pytest.mark.parametrize("keep_state", [True, False])
    def test_sequence_offset(self, device, seed, keep_state):
        """Test that scrambling sequence has no offset (equal likely 0s and 1s)."""
        seq_length = 10000
        batch_size = 100
        s = Scrambler(seed=seed, keep_state=keep_state, binary=True, device=device)
        llr = torch.rand(batch_size, seq_length, device=device)
        s(llr)  # Build scrambler
        # Generate a random sequence
        x = s(torch.zeros_like(llr))
        assert abs(x.float().mean().item() - 0.5) < 0.02

    @pytest.mark.parametrize("keep_state", [True, False])
    def test_sequence_batch(self, device, keep_state):
        """Test that scrambling sequence is random per batch sample."""
        seq_length = 100000
        batch_size = 10
        llr = torch.rand(batch_size, seq_length, device=device)

        s = Scrambler(
            keep_batch_constant=False,
            keep_state=keep_state,
            binary=True,
            device=device,
        )
        x = s(torch.zeros_like(llr))
        # Each batch sample must be different
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                diff_mean = torch.abs(x[i, :] - x[j, :]).float().mean().item()
                assert abs(diff_mean - 0.5) < 0.02

    @pytest.mark.parametrize("keep_state", [True, False])
    def test_sequence_batch_constant(self, device, keep_state):
        """Test that scrambling is the same for all batch samples when
        keep_batch_constant=True."""
        seq_length = 100000
        batch_size = 10
        llr = torch.rand(batch_size, seq_length, device=device)

        s = Scrambler(
            keep_batch_constant=True, keep_state=keep_state, binary=True, device=device
        )
        x = s(torch.zeros_like(llr))
        # Each batch sample is the same
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                assert torch.sum(torch.abs(x[i, :] - x[j, :])).item() == 0

    def test_sequence_realization(self, device):
        """Test that scrambling sequences are random for each new realization."""
        seq_length = 100000
        batch_size = 100
        s = Scrambler(keep_state=False, binary=True, device=device)
        llr = torch.rand(batch_size, seq_length, device=device)
        # Generate random sequences
        x1 = s(torch.zeros_like(llr))
        x2 = s(torch.zeros_like(llr))
        diff_mean = torch.abs(x1 - x2).float().mean().item()
        assert abs(diff_mean - 0.5) < 0.01

    @pytest.mark.parametrize("keep_batch", [True, False])
    def test_inverse_binary(self, device, keep_batch):
        """Test that binary scrambling can be inverted (2x scrambling returns original)."""
        seq_length = 100000
        batch_size = 100

        b = torch.rand(batch_size, seq_length, device=device)
        b = (b > 0.5).float()

        s = Scrambler(
            binary=True, keep_batch_constant=keep_batch, keep_state=True, device=device
        )
        x = s(b)
        x = s(x)
        assert torch.equal(x, b)

    @pytest.mark.parametrize("keep_batch", [True, False])
    def test_inverse_llr(self, device, keep_batch):
        """Test that LLR scrambling can be inverted (2x scrambling returns original)."""
        seq_length = 100000
        batch_size = 100

        llr = torch.rand(batch_size, seq_length, device=device)

        s = Scrambler(
            binary=False, keep_batch_constant=keep_batch, keep_state=True, device=device
        )
        x = s(llr)
        x = s(x)
        assert torch.allclose(x, llr)

    def test_llr(self, device):
        """Test that scrambling works for soft-values (sign flip)."""
        s = Scrambler(binary=False, seed=12345, device=device)
        b = torch.ones(100, 200, device=device)
        x = s(b)
        s2 = Scrambler(binary=True, seed=12345, device=device)
        res = -2.0 * s2(torch.zeros_like(x)) + 1
        assert torch.equal(x, res)

    def test_keep_state(self, device):
        """Test that keep_state works as expected."""
        seq_length = 100000
        batch_size = 100
        llr = torch.rand(batch_size, seq_length, device=device)

        s = Scrambler(binary=True, keep_state=True, device=device)
        res1 = s(torch.zeros_like(llr))
        res2 = s(torch.zeros_like(llr))
        assert torch.equal(res1, res2)

        # Check that sequence is unique with keep_state=False
        s = Scrambler(binary=True, keep_state=False, device=device)
        s(llr)
        res1 = s(torch.zeros_like(llr))
        s(llr)
        res2 = s(torch.zeros_like(llr))
        assert not torch.equal(res1, res2)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        s = Scrambler(keep_state=True, device=device)

        @torch.compile
        def run_scramble(b):
            return s(b)

        b = torch.ones(100, 200, device=device)
        x1 = run_scramble(b)
        # Again with different batch_size
        b = torch.ones(101, 200, device=device)
        x2 = run_scramble(b)

        assert torch.any(x1 != 1)
        assert torch.any(x2 != 1)

    def test_seed(self, device):
        """Test that seed generates reproducible results."""
        seq_length = 100000
        batch_size = 100
        b = torch.zeros(batch_size, seq_length, device=device)

        s1 = Scrambler(seed=1337, binary=True, keep_state=False, device=device)
        res_s1_1 = s1(b)
        res_s1_2 = s1(b)
        # New realization per call
        assert not torch.equal(res_s1_1, res_s1_2)

        # If keep_state=True, the same seed should lead to the same sequence
        s2 = Scrambler(seed=1337, binary=True, keep_state=True, device=device)
        res_s2_1 = s2(b)
        s3 = Scrambler(seed=1337, device=device)
        res_s3_1 = s3(b)
        assert torch.equal(res_s2_1, res_s3_1)

        # Test that seed can also be provided to call
        seed = 987654
        s9 = Scrambler(seed=45234, keep_state=False, device=device)
        s10 = Scrambler(seed=76543, keep_state=True, device=device)
        x1 = s9(x=b, seed=seed)
        x2 = s9(x=b, seed=seed + 1)
        x3 = s9(x=b, seed=seed)
        x4 = s10(x=b, seed=seed)
        assert not torch.equal(x1, x2)  # Different seed
        assert torch.equal(x1, x3)  # Same seed
        assert torch.equal(x1, x4)  # Same seed

    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "p_scr,dt_scr", [("single", torch.float32), ("double", torch.float64)]
    )
    @pytest.mark.parametrize(
        "p_des,dt_des", [("single", torch.float32), ("double", torch.float64)]
    )
    def test_dtype(self, device, dt_in, p_scr, dt_scr, p_des, dt_des):
        """Test that variable dtypes are supported."""
        seq_length = 10
        batch_size = 100

        b = torch.zeros(batch_size, seq_length, dtype=dt_in, device=device)
        s1 = Scrambler(precision=p_scr, device=device)
        s2 = Descrambler(s1, precision=p_des, device=device)
        x = s1(b)
        y = s2(x)
        assert x.dtype == dt_scr
        assert y.dtype == dt_des

    def test_descrambler(self, device):
        """Test that descrambler works as expected."""
        seq_length = 100
        batch_size = 10

        b = torch.zeros(batch_size, seq_length, device=device)
        s1 = Scrambler(device=device)
        s2 = Descrambler(s1, device=device)
        x = s1(b)
        y = s2(x)
        assert torch.equal(b, y)

        # Check if seed is correctly retrieved from scrambler
        s3 = Scrambler(seed=12345, device=device)
        s4 = Descrambler(s3, device=device)
        x = s3(b)
        y = s4(x)
        assert torch.equal(b, y)

    def test_descrambler_nonbin(self, device):
        """Test that descrambler works with non-binary."""
        seq_length = 100
        batch_size = 10

        b = torch.zeros(batch_size, seq_length, device=device)

        # Scrambler binary, but descrambler non-binary
        scrambler = Scrambler(seed=1235456, binary=True, device=device)
        descrambler = Descrambler(scrambler, binary=False, device=device)
        s = 8764
        y = scrambler(b, seed=s)
        z = descrambler(2 * y - 1, seed=s)  # BPSK
        z = 0.5 * (1 + z)  # Remove BPSK
        assert torch.allclose(b, z)

    @pytest.mark.parametrize("shape", [[10, 123], [123]])
    def test_explicit_sequence(self, device, shape):
        """Test that explicit scrambling sequence can be provided."""
        bs = 10
        seq_length = 123

        seq = torch.ones(shape, device=device)
        x = torch.zeros(bs, seq_length, device=device)
        scrambler1 = Scrambler(seed=1245, sequence=seq, binary=True, device=device)
        y1 = scrambler1(x)

        # For all-zero input, output sequence equals scrambling sequence
        if len(shape) == 1:
            y = y1[0, :]
        else:
            y = y1
        assert torch.equal(seq, y)

        # Check that seed has no influence
        scrambler2 = Scrambler(seed=1323, sequence=seq, binary=True, device=device)
        y2 = scrambler2(x)
        assert torch.equal(y1, y2)

    @pytest.mark.parametrize("binary", [True, False])
    def test_explicit_sequence_descrambler(self, device, binary):
        """Test descrambler with explicit scrambling sequence."""
        bs = 10
        seq_length = 123

        seq_np = generate_prng_seq(seq_length, 42)
        seq = torch.tensor(seq_np, dtype=torch.float32, device=device)

        scrambler = Scrambler(sequence=seq, binary=binary, device=device)
        descrambler = Descrambler(scrambler, binary=binary, device=device)
        x = torch.ones(bs, seq_length, device=device)
        y = scrambler(x)
        y2 = scrambler(x, seed=1337)  # Explicit seed should not have any impact
        z = descrambler(y)

        assert not torch.equal(x, y)
        assert torch.allclose(x, z)
        assert torch.equal(y, y2)


class TestTB5GScrambler:
    """Tests for the TB5GScrambler class."""

    @pytest.mark.parametrize("seq_length", [1, 100, 256, 10000])
    @pytest.mark.parametrize("batch_size", [1, 100])
    def test_sequence_dimension(self, device, seq_length, batch_size):
        """Test against correct dimensions of the sequence."""
        s = TB5GScrambler(device=device)
        llr = torch.rand(batch_size, seq_length, device=device)
        x = s(llr)
        assert list(x.shape) == [batch_size, seq_length]

    def test_sequence_dimension_no_batch(self, device):
        """Test non-batch dimension."""
        s = TB5GScrambler(device=device)
        llr = torch.zeros(100, device=device)
        x = s(llr)
        assert x.shape == (100,)

    def test_sequence_batch(self, device):
        """Test that scrambling sequence is the same for all batch samples."""
        seq_length = 1000
        batch_size = 100

        s = TB5GScrambler(binary=True, device=device)
        x = s(torch.zeros(batch_size, seq_length, device=device))

        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                assert torch.sum(torch.abs(x[i, :] - x[j, :])).item() == 0

    def test_sequence_realization(self, device):
        """Test that scrambling sequences are different for different init values."""
        seq_length = 100
        batch_size = 10
        n_rnti_ref = 1337
        n_id_ref = 42

        s_ref = TB5GScrambler(n_rnti_ref, n_id_ref, binary=True, device=device)
        x_ref = s_ref(torch.zeros(batch_size, seq_length, device=device))

        # Randomly init new scramblers
        for _ in range(10):
            n_rnti = int(config.np_rng.integers(0, 2**16 - 1))
            n_id = int(config.np_rng.integers(0, 2**10 - 1))
            if n_rnti == n_rnti_ref and n_id == n_id_ref:
                continue
            s = TB5GScrambler(n_rnti, n_id, binary=True, device=device)
            x = s(torch.zeros(batch_size, seq_length, device=device))
            assert not torch.equal(x_ref, x)

    def test_inverse_binary(self, device):
        """Test that binary scrambling can be inverted."""
        seq_length = 1000
        batch_size = 100

        b = torch.rand(batch_size, seq_length, device=device)
        b = (b > 0.5).float()
        s = TB5GScrambler(binary=True, device=device)
        x = s(b)
        x = s(x)
        assert torch.equal(x, b)

    def test_inverse_llr(self, device):
        """Test that LLR scrambling can be inverted."""
        seq_length = 1000
        batch_size = 100

        llr = torch.rand(batch_size, seq_length, device=device)
        s = TB5GScrambler(binary=False, device=device)
        x = s(llr)
        x = s(x)
        assert torch.allclose(x, llr)

    def test_torch_compile(self, device):
        """Test that torch.compile works as expected."""
        s = TB5GScrambler(device=device)

        @torch.compile
        def run_scramble(b):
            return s(b)

        b = torch.ones(10, 200, device=device)
        x1 = run_scramble(b)
        # Again with different batch_size
        b = torch.ones(11, 200, device=device)
        x2 = run_scramble(b)

        assert torch.any(x1 != 1)
        assert torch.any(x2 != 1)

    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "p_scr,dt_scr", [("single", torch.float32), ("double", torch.float64)]
    )
    @pytest.mark.parametrize(
        "p_des,dt_des", [("single", torch.float32), ("double", torch.float64)]
    )
    def test_dtype(self, device, dt_in, p_scr, dt_scr, p_des, dt_des):
        """Test that variable dtypes are supported."""
        seq_length = 10
        batch_size = 100

        b = torch.zeros(batch_size, seq_length, dtype=dt_in, device=device)
        s1 = TB5GScrambler(precision=p_scr, device=device)
        s2 = Descrambler(s1, precision=p_des, device=device)
        x = s1(b)
        y = s2(x)
        assert x.dtype == dt_scr
        assert y.dtype == dt_des

    def test_descrambler(self, device):
        """Test that descrambler works as expected."""
        seq_length = 100
        batch_size = 10

        b = torch.zeros(batch_size, seq_length, device=device)
        s1 = TB5GScrambler(device=device)
        s2 = Descrambler(s1, device=device)
        x = s1(b)
        y = s2(x)
        assert torch.equal(b, y)

    def test_scrambler_binary(self, device):
        """Test that binary flag can be used as input."""
        seq_length = 100
        batch_size = 10

        b = torch.ones(batch_size, seq_length, device=device)
        scrambler = TB5GScrambler(binary=True, device=device)

        x1 = scrambler(b)  # Binary scrambling
        x2 = scrambler(b)  # Binary scrambling
        x3 = scrambler(b, binary=True)  # Binary scrambling
        x4 = scrambler(b, binary=False)  # Non-binary scrambling

        assert torch.equal(x1, x2)
        assert torch.equal(x2, x3)
        assert torch.allclose(x1, 0.5 * (1 + x4))

    def test_5gnr_reference(self, device):
        """Test against 5G NR reference."""
        bs = 2
        length = 100

        # Check valid inputs
        n_rs = [0, 10, 65535]
        n_ids = [0, 10, 1023]
        s_old = None
        for n_r in n_rs:
            for n_id in n_ids:
                s = TB5GScrambler(n_id=n_id, n_rnti=n_r, device=device)(
                    torch.zeros(bs, length, device=device)
                )
                if s_old is not None:
                    assert not torch.equal(s, s_old)
                s_old = s

        # Test against reference example
        n_rnti = 20001
        n_id = 41
        s_ref = np.array(
            [
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            ]
        )
        s = TB5GScrambler(n_id=n_id, n_rnti=n_rnti, device=device)(
            torch.zeros(1, length, device=device)
        )
        s = s.squeeze(0).cpu().numpy()
        assert np.array_equal(s, s_ref)

        # And test against wrong parameters
        s = TB5GScrambler(n_id=n_id, n_rnti=n_rnti + 1, device=device)(
            torch.zeros(1, length, device=device)
        )
        s = s.squeeze(0).cpu().numpy()
        assert not np.array_equal(s, s_ref)

        # Test that PUSCH and PDSCH are the same for single cw mode
        s_ref = TB5GScrambler(
            n_id=n_id, n_rnti=n_rnti, channel_type="PUSCH", device=device
        )(torch.zeros(1, length, device=device))

        # cw_idx has no impact in uplink
        s = TB5GScrambler(
            n_id=n_id,
            n_rnti=n_rnti,
            codeword_index=1,
            channel_type="PUSCH",
            device=device,
        )(torch.zeros(1, length, device=device))
        assert torch.equal(s_ref, s)

        # Downlink equals uplink for cw_idx=0
        s = TB5GScrambler(
            n_id=n_id,
            n_rnti=n_rnti,
            codeword_index=0,
            channel_type="PDSCH",
            device=device,
        )(torch.zeros(1, length, device=device))
        assert torch.equal(s_ref, s)

        # Downlink is different from uplink for cw_idx=1
        s = TB5GScrambler(
            n_id=n_id,
            n_rnti=n_rnti,
            codeword_index=1,
            channel_type="PDSCH",
            device=device,
        )(torch.zeros(1, length, device=device))
        assert not torch.equal(s_ref, s)

    @pytest.mark.parametrize("n_r", [-1, 1.2, 65536])
    @pytest.mark.parametrize("n_id", [0, 10, 1023])
    def test_5gnr_invalid_n_rnti(self, device, n_r, n_id):
        """Test that invalid n_rnti values raise ValueError."""
        with pytest.raises(ValueError):
            TB5GScrambler(n_id=n_id, n_rnti=n_r, device=device)

    @pytest.mark.parametrize("n_r", [0, 10, 65535])
    @pytest.mark.parametrize("n_id", [-1, 1.2, 1024])
    def test_5gnr_invalid_n_id(self, device, n_r, n_id):
        """Test that invalid n_id values raise ValueError."""
        with pytest.raises(ValueError):
            TB5GScrambler(n_id=n_id, n_rnti=n_r, device=device)

    def test_multi_user(self, device):
        """Test multi-stream functionality."""
        seq_length = 1000
        batch_size = 13

        n_rntis = [1, 38282, 1337, 36443]
        n_ids = [123, 42, 232, 134]

        u = torch.zeros(batch_size, 2, len(n_rntis), seq_length, device=device)
        s_ref = torch.zeros_like(u)

        # Generate batch of multiple streams individually
        for idx, (n_rnti, n_id) in enumerate(zip(n_rntis, n_ids)):
            scrambler = TB5GScrambler(n_id=n_id, n_rnti=n_rnti, device=device)
            s_ref[..., idx, :] = scrambler(u[..., 0, :])

        # Run scrambler one-shot with list of n_rnti/n_id
        scrambler = TB5GScrambler(n_id=n_ids, n_rnti=n_rntis, device=device)
        s = scrambler(u)

        # Scrambling sequences should be equivalent
        assert torch.equal(s, s_ref)

        # Also test descrambler
        u_hat = Descrambler(scrambler, device=device)(s)
        assert torch.equal(u_hat, torch.zeros_like(u_hat))


class TestDescrambler:
    """Tests for the Descrambler class."""

    def test_invalid_scrambler(self):
        """Test that descrambler raises error for invalid scrambler."""
        with pytest.raises(TypeError):
            Descrambler("not_a_scrambler")

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        scrambler = Scrambler(seed=42, keep_state=True)
        descrambler = Descrambler(scrambler, binary=False)

        llrs = torch.randn(10, 100, device=scrambler.device)
        scrambled = scrambler(llrs, binary=False)
        unscrambled = descrambler(scrambled)
        assert torch.allclose(llrs, unscrambled)

