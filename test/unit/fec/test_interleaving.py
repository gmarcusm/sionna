#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.interleaving module."""

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fec.interleaving import (
    RandomInterleaver,
    RowColumnInterleaver,
    Deinterleaver,
    Turbo3GPPInterleaver,
)
from sionna.phy.fec.scrambling import Scrambler


class TestRandomInterleaver:
    """Tests for RandomInterleaver class."""

    @pytest.mark.parametrize("keep_batch", [True, False])
    @pytest.mark.parametrize("inverse", [True, False])
    @pytest.mark.parametrize("seq_length", [1, 100, 256, 1000])
    @pytest.mark.parametrize("batch_size", [1, 100, 256, 1000])
    def test_sequence_dimension(
        self, device, keep_batch, inverse, seq_length, batch_size
    ):
        """Test against correct dimensions of the sequence."""
        inter = RandomInterleaver(
            keep_batch_constant=keep_batch, inverse=inverse, device=device
        )
        x = inter(torch.zeros(batch_size, seq_length, device=device))
        assert x.shape == (batch_size, seq_length)

    @pytest.mark.parametrize("keep_batch", [True, False])
    def test_inverse(self, device, keep_batch):
        """Test that inverse permutation matches to permutation."""
        seq_length = 1000
        batch_size = 100

        inter = RandomInterleaver(
            keep_batch_constant=keep_batch, seed=123, device=device
        )
        inter2 = RandomInterleaver(
            keep_batch_constant=keep_batch, inverse=True, seed=123, device=device
        )

        x = torch.arange(seq_length, device=device, dtype=torch.float32)
        x = x.unsqueeze(0).expand(batch_size, -1)

        y = inter(x)
        z = inter2(y)

        for i in range(batch_size):
            expected = torch.arange(seq_length, device=device, dtype=torch.float32)
            assert torch.allclose(z[i, :], expected)

        # Also test explicit seed
        y = inter(x, seed=12345)
        z = inter2(y, seed=12345)

        for i in range(batch_size):
            expected = torch.arange(seq_length, device=device, dtype=torch.float32)
            assert torch.allclose(z[i, :], expected)

    def test_inverse_no_batch_dim(self, device):
        """Test inverse interleaving without batch dimension."""
        seq_length = 1000

        inter = RandomInterleaver(keep_batch_constant=True, seed=42, device=device)
        deinter = Deinterleaver(inter)

        x = torch.arange(seq_length, device=device, dtype=torch.float32)
        y = inter(x)
        z = deinter(y)

        assert not torch.allclose(x, y)
        assert torch.allclose(x, z)

    def test_sequence_batch(self, device):
        """Test that interleaver sequence is random per batch sample.

        Remark: this test must fail for keep_batch_constant=True.
        """
        seq_length = 1000
        batch_size = 10

        i1 = RandomInterleaver(keep_batch_constant=False, device=device)
        i2 = RandomInterleaver(keep_batch_constant=True, device=device)

        x = torch.arange(seq_length, device=device, dtype=torch.float32)
        x = x.unsqueeze(0).expand(batch_size, -1)

        y1 = i1(x)
        y2 = i2(x)

        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                # Different sequences per batch for keep_batch_constant=False
                assert not torch.allclose(y1[i, :], y1[j, :])
                # Same sequences per batch for keep_batch_constant=True
                assert torch.allclose(y2[i, :], y2[j, :])

    @pytest.mark.parametrize("keep_batch", [True, False])
    def test_sequence_realization(self, device, keep_batch):
        """Test that interleaver sequences are random for each new realization
        iff keep_state==False.
        """
        seq_length = 1000
        batch_size = 10

        inter = RandomInterleaver(
            keep_batch_constant=keep_batch, keep_state=True, device=device
        )

        x = torch.arange(seq_length, device=device, dtype=torch.float32)
        x = x.unsqueeze(0).expand(batch_size, -1)

        # Same results if keep_state=True
        x1 = inter(x)
        x2 = inter(x)
        assert torch.allclose(x1, x2)

        inter2 = RandomInterleaver(
            keep_batch_constant=keep_batch, keep_state=False, device=device
        )
        # Different results if keep_state=False
        x1 = inter2(x)
        x2 = inter2(x)
        assert not torch.allclose(x1, x2)

    @pytest.mark.parametrize("keep_batch", [True, False])
    @pytest.mark.parametrize(
        "shape,axis",
        [
            (shape, -1 if a == 0 else a)
            for shape in [
                [1000],
                [10, 20, 30],
                [10, 22, 33, 44],
                [20, 10, 10, 10, 6],
            ]
            for a in range(len(shape))
        ],
    )
    def test_multi_dim(self, device, keep_batch, shape, axis):
        """Test that 2+D Tensors permutation can be inverted/removed.

        Inherently tests that the output dimensions match.
        """
        llr = torch.rand(shape, device=device) * 200 - 100

        inter = RandomInterleaver(
            keep_batch_constant=keep_batch,
            axis=axis,
            keep_state=True,
            device=device,
        )
        inter2 = RandomInterleaver(
            keep_batch_constant=keep_batch,
            axis=axis,
            keep_state=True,
            inverse=True,
            device=device,
        )

        x = inter(llr, seed=1234)
        # After interleaving arrays must be different
        assert not torch.allclose(x, llr)

        # After deinterleaving arrays should be equal again
        x = inter2(x, seed=1234)
        assert torch.allclose(x, llr)

    @pytest.mark.parametrize(
        "shape", [[10, 20, 30], [10, 22, 33, 44], [20, 10, 10, 10, 6]]
    )
    def test_invalid_shapes(self, device, shape):
        """Test that invalid shapes/axis parameter raise error."""
        with pytest.raises(ValueError):
            # Axis out of bounds must raise error
            inter = RandomInterleaver(axis=len(shape), device=device)
            llr = torch.rand(shape, device=device)
            inter(llr)

    @pytest.mark.parametrize("keep_batch", [True, False])
    def test_seed(self, device, keep_batch):
        """Test that seed can be fed.

        Remark: this test generates multiple interleavers to test the
        influence of different seeds.
        """
        seq_length = 1000
        batch_size = 10
        seed = 123456

        i1 = RandomInterleaver(
            keep_batch_constant=keep_batch,
            seed=seed,
            keep_state=True,
            device=device,
        )
        i2 = RandomInterleaver(
            keep_batch_constant=keep_batch, keep_state=True, device=device
        )
        i3 = RandomInterleaver(
            keep_batch_constant=keep_batch,
            seed=seed,
            keep_state=True,
            device=device,
        )
        i4 = RandomInterleaver(
            keep_batch_constant=keep_batch,
            seed=seed + 1,
            keep_state=True,
            device=device,
        )

        x = torch.arange(seq_length, device=device, dtype=torch.float32)
        x = x.unsqueeze(0).expand(batch_size, -1)

        x1 = i1(x)
        x2 = i2(x)
        x3 = i3(x)
        x4 = i4(x)

        # x1 and x3 must be the same (same seed)
        assert torch.allclose(x1, x3)

        # x1 and x2/x4 are not the same (different seed)
        assert not torch.allclose(x1, x2)
        assert not torch.allclose(x1, x4)

        # Test that seed can be also provided to call
        test_seed = 987654
        i11 = RandomInterleaver(
            keep_batch_constant=keep_batch,
            seed=seed,
            keep_state=False,
            device=device,
        )
        i21 = RandomInterleaver(
            keep_batch_constant=keep_batch,
            keep_state=False,
            inverse=True,
            device=device,
        )

        x7 = i11(x, seed=test_seed)
        x8 = i11(x, seed=test_seed + 1)
        x9 = i11(x, seed=test_seed)
        x10 = i1(x, seed=test_seed)

        assert not torch.allclose(x7, x8)  # Different seed
        assert torch.allclose(x7, x9)  # Same seed
        assert torch.allclose(x7, x10)  # Same seed (keep_state=False)

        # Test that random seed allows inverse
        x11 = i11(x, seed=test_seed)
        # Use different interleaver with same seed to de-interleave
        x12 = i21(x11, seed=test_seed)
        assert torch.allclose(x, x12)  # Identity

    @pytest.mark.parametrize("seed", range(50))
    def test_s_param(self, device, seed):
        """Test that interleaver outputs correct S parameter for given seed."""
        k = 100
        inter = RandomInterleaver(device=device)

        x = torch.arange(k, device=device, dtype=torch.float32).unsqueeze(0)
        x_int = inter(x, seed=seed).squeeze(0).cpu().numpy()

        s_inter = inter.find_s_min(seed=seed, seq_length=k)

        # Verify S parameter
        cont = True
        for s_min in range(1, k):
            for i in range(k):
                a = x_int[i]
                if i - s_min >= 0:
                    b = x_int[i - s_min]
                    if np.abs(a - b) <= s_min:
                        cont = False
                if i + s_min < k:
                    b = x_int[i + s_min]
                    if np.abs(a - b) <= s_min:
                        cont = False
            if not cont:
                break

        assert s_inter == s_min

    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    def test_dtype(self, device, precision, dt_in):
        """Test that variable dtypes are supported."""
        seq_length = 10
        batch_size = 100

        dtypes_supported = {
            "single": torch.float32,
            "double": torch.float64,
        }
        dt_out = dtypes_supported[precision]

        b = torch.zeros(batch_size, seq_length, dtype=dt_in, device=device)
        inter = RandomInterleaver(precision=precision, device=device)
        x = inter(b)
        assert x.dtype == dt_out

    @pytest.mark.parametrize("shape", [[10, 20, 30], [10, 22, 33, 44]])
    def test_torch_compile(self, device, shape):
        """Test that torch.compile works as expected."""
        llr = torch.rand(shape, device=device)
        inter = RandomInterleaver(keep_batch_constant=True, device=device)

        compiled_inter = torch.compile(inter)
        x1 = compiled_inter(llr)
        x2 = compiled_inter(llr)

        # After interleaving arrays must be different
        assert not torch.allclose(x1, llr)
        assert torch.allclose(x1, x2)


class TestRowColumnInterleaver:
    """Tests for RowColumnInterleaver class."""

    @pytest.mark.parametrize("depth", [1, 2, 4, 7, 8])
    @pytest.mark.parametrize("seq_length", [1, 100, 256, 1000])
    def test_sequence_dimension(self, device, depth, seq_length):
        """Test against correct dimensions of the perm sequence."""
        inter = RowColumnInterleaver(row_depth=depth, device=device)
        x, y = inter._generate_perm_rc(seq_length, depth)
        assert x.shape[0] == seq_length
        assert y.shape[0] == seq_length

    @pytest.mark.parametrize("depth", [1, 2, 4, 7, 8])
    @pytest.mark.parametrize("case", [(100, 9), (100, 11)])
    def test_dimension(self, device, depth, case):
        """Test against dimension mismatches."""
        seq_length = 10
        batch_size = 100

        inter = RowColumnInterleaver(row_depth=depth, device=device)
        llr = torch.rand(batch_size, seq_length, device=device)
        inter(llr)

        llr = torch.rand(int(case[0]), int(case[1]), device=device)
        # Should run without error
        inter(llr)

    @pytest.mark.parametrize("depth", [1, 2, 4, 7, 8])
    def test_inverse(self, device, depth):
        """Test that permutation can be inverted/removed."""
        seq_length = 1000
        batch_size = 100

        llr = torch.rand(batch_size, seq_length, device=device)

        inter = RowColumnInterleaver(row_depth=depth, device=device)
        inter2 = RowColumnInterleaver(row_depth=depth, inverse=True, device=device)

        x = inter(llr)
        y = inter2(x)
        assert torch.allclose(y, llr)

    def test_inverse_no_batch_dim(self, device):
        """Test inverse without batch dimension."""
        inter = RowColumnInterleaver(row_depth=3, device=device)
        deinter = Deinterleaver(inter)

        x = torch.arange(100, device=device, dtype=torch.float32)
        y = inter(x)
        z = deinter(y)

        assert not torch.allclose(x, y)
        assert torch.allclose(x, z)

    @pytest.mark.parametrize("depth", [2, 4, 7, 8])
    @pytest.mark.parametrize(
        "shape,axis",
        [
            (shape, a)
            for shape in [
                [1000],
                [10, 20, 30],
                [10, 22, 33, 44],
                [20, 10, 10, 10, 9],
            ]
            for a in range(len(shape))
        ],
    )
    def test_multi_dim(self, device, depth, shape, axis):
        """Test that 2+D Tensors permutation can be inverted/removed.

        Inherently tests that the output dimensions match.
        Also tests that arrays are different.
        """
        llr = torch.rand(shape, device=device)

        inter = RowColumnInterleaver(row_depth=depth, axis=axis, device=device)
        inter2 = RowColumnInterleaver(
            row_depth=depth, axis=axis, inverse=True, device=device
        )

        x = inter(llr)
        # After interleaving arrays must be different
        assert not torch.allclose(x, llr)

        # After deinterleaving it should be equal again
        x = inter2(x)
        assert torch.allclose(x, llr)

    @pytest.mark.parametrize(
        "shape", [[10, 20, 30], [10, 22, 33, 44], [20, 10, 10, 10, 6]]
    )
    def test_invalid_axis(self, device, shape):
        """Test that 2+D Tensors and invalid axis raise error."""
        with pytest.raises(ValueError):
            inter = RowColumnInterleaver(row_depth=4, axis=len(shape), device=device)
            llr = torch.rand(shape, device=device)
            inter(llr)

    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    def test_dtype(self, device, precision, dt_in):
        """Test that variable dtypes are supported."""
        seq_length = 10
        batch_size = 100

        dtypes_supported = {
            "single": torch.float32,
            "double": torch.float64,
        }
        dt_out = dtypes_supported[precision]

        b = torch.zeros(batch_size, seq_length, dtype=dt_in, device=device)
        inter = RowColumnInterleaver(
            row_depth=4, precision=precision, device=device
        )
        x = inter(b)
        assert x.dtype == dt_out

    @pytest.mark.parametrize("depth", [2, 4, 7, 8])
    @pytest.mark.parametrize(
        "shape,axis",
        [
            (shape, a)
            for shape in [[10, 20, 30], [10, 22, 33, 44]]
            for a in range(len(shape))
        ],
    )
    def test_torch_compile(self, device, depth, shape, axis):
        """Test that torch.compile works as expected."""
        llr = torch.rand(shape, device=device)

        inter = RowColumnInterleaver(row_depth=depth, axis=axis, device=device)

        compiled_inter = torch.compile(inter)
        x1 = compiled_inter(llr)
        x2 = compiled_inter(llr)

        # After interleaving arrays must be different
        assert not torch.allclose(x1, llr)
        assert torch.allclose(x1, x2)


class TestDeinterleaver:
    """Tests for Deinterleaver class."""

    @pytest.mark.parametrize("keep_batch", [True, False])
    @pytest.mark.parametrize("seed", [None, 1234, 876])
    def test_identity(self, device, keep_batch, seed):
        """Test that deinterleave can invert Random-/RCInterleaver."""
        seq_length = 10
        batch_size = 100

        x = torch.rand(batch_size, seq_length, device=device)

        # Test RowColumnInterleaver
        inter_rc = RowColumnInterleaver(row_depth=3, device=device)
        deinter_rc = Deinterleaver(inter_rc)

        y = inter_rc(x)
        z = deinter_rc(y)

        assert not torch.allclose(x, y)
        assert torch.allclose(x, z)

        # Test RandomInterleaver
        inter_random = RandomInterleaver(
            keep_batch_constant=keep_batch, seed=seed, device=device
        )
        deinter_random = Deinterleaver(inter_random)

        y = inter_random(x)
        z = deinter_random(y)

        assert not torch.allclose(x, y)
        assert torch.allclose(x, z)

    @pytest.mark.parametrize("axis", [1, 2, 3, -1, -2])
    def test_axis(self, device, axis):
        """Test that deinterleaver operates on correct axis."""
        x = torch.rand(10, 20, 20, 20, device=device)

        # Test RowColumnInterleaver
        inter_rc = RowColumnInterleaver(row_depth=3, axis=axis, device=device)
        deinter_rc = Deinterleaver(inter_rc)

        y = inter_rc(x)
        z = deinter_rc(y)

        assert not torch.allclose(x, y)
        assert torch.allclose(x, z)

        # Test RandomInterleaver
        inter_random = RandomInterleaver(axis=axis, device=device)
        deinter_random = Deinterleaver(inter_random)

        y = inter_random(x)
        z = deinter_random(y)

        assert not torch.allclose(x, y)
        assert torch.allclose(x, z)

    @pytest.mark.parametrize(
        "dt_in",
        [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.complex64,
            torch.complex128,
        ],
    )
    def test_dtype(self, device, precision, dt_in):
        """Test that arbitrary dtypes are supported."""
        if dt_in.is_complex:
            x_real = torch.rand(10, 20, device=device)
            x = torch.complex(x_real, torch.zeros_like(x_real))
            x = x.to(dt_in)
        else:
            x = torch.rand(10, 20, device=device).to(dt_in)

        # Test RowColumnInterleaver
        inter_rc = RowColumnInterleaver(
            row_depth=3, precision=precision, device=device
        )
        deinter_rc1 = Deinterleaver(inter_rc)
        deinter_rc2 = Deinterleaver(inter_rc, precision=precision)

        y = inter_rc(x)
        z1 = deinter_rc1(y)
        z2 = deinter_rc2(y)

        assert z1.dtype == y.dtype
        assert z2.dtype == y.dtype

        # Test RandomInterleaver
        inter_rand = RandomInterleaver(precision=precision, device=device)
        deinter_rand1 = Deinterleaver(inter_rand)
        deinter_rand2 = Deinterleaver(inter_rand, precision=precision)

        y = inter_rand(x)
        z1 = deinter_rand1(y)
        z2 = deinter_rand2(y)

        assert z1.dtype == y.dtype
        assert z2.dtype == y.dtype

    def test_invalid_input(self, device):
        """Test against invalid parameters."""
        inter1 = RandomInterleaver(device=device)
        scram = Scrambler(device=device)

        # Invalid input
        for s in (scram, None, 124):
            with pytest.raises(ValueError):
                Deinterleaver(s)

    @pytest.mark.parametrize("depth", [2, 4, 7, 8])
    @pytest.mark.parametrize(
        "shape,axis",
        [
            (shape, a)
            for shape in [[10, 20, 30], [10, 22, 33, 44]]
            for a in range(len(shape))
        ],
    )
    def test_torch_compile(self, device, depth, shape, axis):
        """Test that torch.compile works as expected."""
        llr = torch.rand(shape, device=device)

        int_rc = RowColumnInterleaver(row_depth=depth, axis=axis, device=device)
        int_rand = RandomInterleaver(device=device)

        de_int_rc = Deinterleaver(int_rc)
        de_int_rand = Deinterleaver(int_rand)

        # Compile deinterleavers
        compiled_deint_rc = torch.compile(de_int_rc)
        compiled_deint_rand = torch.compile(de_int_rand)

        x1 = compiled_deint_rc(int_rc(llr))
        x2 = compiled_deint_rand(int_rand(llr))

        # After interleaving+deinterleaving arrays should match
        assert torch.allclose(llr, x1)
        assert torch.allclose(llr, x2)


class TestTurbo3GPPInterleaver:
    """Tests for Turbo3GPPInterleaver class."""

    @pytest.mark.parametrize("inverse", [True, False])
    @pytest.mark.parametrize("seq_length", [1, 100, 256, 1000])
    @pytest.mark.parametrize("batch_size", [1, 100, 256, 1000])
    def test_sequence_dimension(self, device, inverse, seq_length, batch_size):
        """Test against correct dimensions of the sequence."""
        inter = Turbo3GPPInterleaver(inverse=inverse, device=device)
        x = inter(torch.zeros(batch_size, seq_length, device=device))
        assert x.shape == (batch_size, seq_length)

    def test_inverse(self, device):
        """Test that inverse permutation matches to permutation."""
        seq_length = 1000
        batch_size = 100

        inter = Turbo3GPPInterleaver(device=device)
        inter2 = Turbo3GPPInterleaver(inverse=True, device=device)
        deinter = Deinterleaver(inter)

        x = torch.arange(seq_length, device=device, dtype=torch.float32)
        x = x.unsqueeze(0).expand(batch_size, -1)

        y = inter(x)
        z = inter2(y)
        z2 = deinter(y)

        for i in range(batch_size):
            expected = torch.arange(seq_length, device=device, dtype=torch.float32)
            assert torch.allclose(z[i, :], expected)
            assert torch.allclose(z2[i, :], expected)

    def test_inverse_no_batch_dim(self, device):
        """Test inverse without batch dimension."""
        seq_length = 1000

        inter = Turbo3GPPInterleaver(device=device)
        deinter = Deinterleaver(inter)

        x = torch.arange(seq_length, device=device, dtype=torch.float32)
        y = inter(x)
        z = deinter(y)

        assert not torch.allclose(x, y)
        assert torch.allclose(x, z)

    @pytest.mark.parametrize("case", [(102, 10), (101, 11)])
    def test_dimension(self, device, case):
        """Test that dimensions can be changed."""
        seq_length = 10
        batch_size = 100

        inter = Turbo3GPPInterleaver(device=device)
        inter(torch.rand(batch_size, seq_length, device=device))

        llr = torch.rand(int(case[0]), int(case[1]), device=device)
        inter(llr)

    @pytest.mark.parametrize(
        "shape,axis",
        [
            (shape, -1 if a == 0 else a)
            for shape in [[10, 20, 30], [10, 22, 33, 44], [20, 10, 10, 10, 6]]
            for a in range(len(shape))
        ],
    )
    def test_multi_dim(self, device, shape, axis):
        """Test that 2+D Tensors permutation can be inverted/removed.

        Inherently tests that the output dimensions match.
        """
        llr = torch.rand(shape, device=device) * 200 - 100

        inter = Turbo3GPPInterleaver(axis=axis, device=device)
        inter2 = Turbo3GPPInterleaver(axis=axis, inverse=True, device=device)

        x = inter(llr)
        # After interleaving arrays must be different
        assert not torch.allclose(x, llr)

        # After deinterleaving arrays should be equal again
        x = inter2(x)
        assert torch.allclose(x, llr)

    def test_invalid_shapes(self, device):
        """Test that invalid shapes/axis parameter raise error."""
        # k > 6144
        inter = Turbo3GPPInterleaver(axis=-1, device=device)
        s = [10, 6145]
        llr = torch.rand(s, device=device)
        with pytest.raises(ValueError):
            inter(llr)

        # Axis out of bounds
        s = [10, 20, 30]
        with pytest.raises(ValueError):
            inter2 = Turbo3GPPInterleaver(axis=len(s), device=device)
            llr = torch.rand(s, device=device)
            inter2(llr)

    @pytest.mark.parametrize("dt_in", [torch.float32, torch.float64])
    def test_dtype(self, device, precision, dt_in):
        """Test that variable dtypes are supported."""
        seq_length = 10
        batch_size = 100

        dtypes_supported = {
            "single": torch.float32,
            "double": torch.float64,
        }
        dt_out = dtypes_supported[precision]

        b = torch.zeros(batch_size, seq_length, dtype=dt_in, device=device)
        inter = Turbo3GPPInterleaver(precision=precision, device=device)
        x = inter(b)
        assert x.dtype == dt_out

    @pytest.mark.parametrize(
        "shape", [[10, 20, 30], [10, 22, 33, 44], [20, 10, 10, 10, 9]]
    )
    def test_torch_compile(self, device, shape):
        """Test that torch.compile works as expected."""
        llr = torch.rand(shape, device=device)
        inter = Turbo3GPPInterleaver(device=device)

        compiled_inter = torch.compile(inter)
        x1 = compiled_inter(llr)
        x2 = compiled_inter(llr)

        # After interleaving arrays must be different
        assert not torch.allclose(x1, llr)

        # XLA and graph mode should result in the same array
        assert torch.allclose(x1, x2)

    def test_torch_compile_variable_lengths(self, device):
        """Test that torch.compile works with variable input lengths."""
        inter = Turbo3GPPInterleaver(device=device)
        compiled_inter = torch.compile(inter)
        llr = torch.rand(10, 100, device=device)
        x = compiled_inter(llr)
        llr = torch.rand(10, 101, device=device)
        x = compiled_inter(llr)
