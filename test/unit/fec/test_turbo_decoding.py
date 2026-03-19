#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.phy.fec.turbo.decoding module."""

import os

import numpy as np
import pytest
import torch

from sionna.phy.fec.turbo import TurboEncoder, TurboDecoder
from sionna.phy.mapping import BinarySource
from sionna.phy.fec.conv.decoding import BCJRDecoder


# Get test data directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(TEST_DIR, "..", "..", "codes", "turbo")


class TestTurboDecoder:
    """Tests for TurboDecoder class."""

    @pytest.mark.parametrize("t", [False, True])
    @pytest.mark.parametrize("k", [10, 20, 50, 100])
    @pytest.mark.parametrize("rate", [1 / 2, 1 / 3])
    def test_output_dim_num_stab(self, device, rate, k, t):
        """Test that output dims are correct (=k) and numerical stability."""
        bs = 10
        cl = 5  # constraint length

        n = int(k / rate)
        if t:
            n += int(cl / rate)

        enc = TurboEncoder(
            rate=rate, constraint_length=cl, terminate=t, device=device
        )
        dec = TurboDecoder(enc, num_iter=1, hard_out=False, device=device)

        u = torch.zeros(bs, k, dtype=torch.float32, device=device)
        c = enc(u)

        # Input to decoder is channel symbols (as LLR)
        y = torch.zeros(bs, c.shape[-1], device=device)
        u_hat = dec(y)

        # Check that output has the correct shape
        assert u_hat.shape == torch.Size([bs, k])

        # Check numerical stability
        assert torch.all(torch.isfinite(u_hat))

    @pytest.mark.parametrize("t", [False, True])
    @pytest.mark.parametrize("k", [10, 20, 50, 100])
    @pytest.mark.parametrize("rate", [1 / 2, 1 / 3])
    def test_identity(self, device, rate, k, t):
        """Test that encoded all-zero codeword yields all-zero estimates,
        when SNR is perfect (high confidence LLRs)."""
        bs = 10
        cl = 5  # constraint length

        enc = TurboEncoder(
            rate=rate, constraint_length=cl, terminate=t, device=device
        )
        dec = TurboDecoder(enc, num_iter=4, hard_out=True, device=device)

        u = torch.zeros(bs, k, dtype=torch.float32, device=device)
        c = enc(u)

        # Perfect channel: high confidence LLRs (logit convention)
        # 0 -> -large, 1 -> +large
        y = 20.0 * (2.0 * c - 1.0)  # High SNR LLRs

        u_hat = dec(y)

        # All-zero u should yield all-zero u_hat
        assert torch.allclose(u, u_hat)

    @pytest.mark.parametrize("s_", [[4, 5, 5], []])
    def test_multi_dimensional(self, device, s_):
        """Test against arbitrary shapes."""
        k = 120
        rate = 1 / 2

        source = BinarySource(device=device)
        enc = TurboEncoder(
            rate=rate, constraint_length=5, terminate=False, device=device
        )
        dec = TurboDecoder(enc, num_iter=4, hard_out=True, device=device)

        s = s_.copy()
        bs = int(np.prod(s)) if s else 1
        b = source([bs, k])

        if s:
            s.append(k)
            b_res = b.reshape(s)
        else:
            b_res = b.reshape(k)

        # Encode
        c = enc(b)

        # Create high confidence LLRs
        y = 20.0 * (2.0 * c - 1.0)

        if s:
            s_out = s.copy()
            s_out[-1] = c.shape[-1]
            y_res = y.reshape(s_out)
        else:
            y_res = y.reshape(c.shape[-1])

        # Decode 2D tensor
        u_hat = dec(y)
        # Decode multi-D tensor
        u_hat_res = dec(y_res)

        # Test that shape was preserved
        expected_shape = list(b_res.shape)
        assert list(u_hat_res.shape) == expected_shape

        # And reshape to 2D shape
        u_hat_res_flat = (
            u_hat_res.reshape(bs, k) if s else u_hat_res.reshape(1, k)
        )
        # Both versions should yield same result
        assert torch.equal(u_hat, u_hat_res_flat)

    def test_batch(self, device):
        """Test that all samples in batch yield same output (for same input)."""
        bs = 100
        k = 120

        source = BinarySource(device=device)
        enc = TurboEncoder(rate=0.5, constraint_length=6, device=device)
        dec = TurboDecoder(enc, num_iter=4, hard_out=True, device=device)

        b = source([1, 15, k])
        b_rep = b.repeat(bs, 1, 1)

        c = enc(b_rep)
        y = 20.0 * (2.0 * c - 1.0)

        u_hat = dec(y)

        for i in range(bs):
            assert torch.equal(u_hat[0, :, :], u_hat[i, :, :])

    def test_ber_match(self, device):
        """Test that BER is within expected bounds for a given SNR.

        Note: This test may be slow on CPU. Skipped on CPU.
        """
        if device == "cpu":
            pytest.skip("BER test skipped on CPU due to performance")

        bs = 100
        k = 40
        snr_db = 3.0
        target_ber_max = 0.02  # Should be below this for SNR=3dB

        source = BinarySource(device=device)
        enc = TurboEncoder(
            rate=1 / 3, constraint_length=4, terminate=True, device=device
        )
        dec = TurboDecoder(enc, num_iter=6, hard_out=True, device=device)

        # SNR setup
        snr_lin = 10 ** (snr_db / 10)
        noise_var = 1 / (2 * snr_lin)
        noise_std = np.sqrt(noise_var)

        u = source([bs, k])
        c = enc(u)

        # BPSK modulation: 0 -> -1, 1 -> +1 (using logit convention)
        x = 2.0 * c - 1.0

        # Add AWGN
        noise = torch.randn_like(x) * noise_std
        y_channel = x + noise

        # Compute LLRs (logit convention: log p(1)/p(0))
        llr = 2.0 * y_channel / noise_var

        u_hat = dec(llr)

        # Calculate BER
        ber = (u != u_hat).float().mean().item()

        # Check that BER is below threshold
        assert ber < target_ber_max, f"BER {ber:.4f} is above threshold {target_ber_max}"

    @pytest.mark.parametrize("k", [40, 112, 168, 432])
    def test_ref_implementation(self, device, k):
        """Test against pre-decoded outputs from reference implementation."""
        if not os.path.exists(REF_PATH):
            pytest.skip("Reference data not found")

        # Reference data was generated with r=1/3, Eb/N0=0dB
        r = 1 / 3
        ebno = 0.0
        no = 1 / (r * (10 ** (-ebno / 10)))  # no = 3.0

        y_path = os.path.join(REF_PATH, f"ref_k{k}_y.npy")
        uhat_path = os.path.join(REF_PATH, f"ref_k{k}_uhat.npy")

        if not os.path.exists(y_path) or not os.path.exists(uhat_path):
            pytest.skip(f"Reference data for k={k} not found")

        yref = np.load(y_path)
        uref = np.load(uhat_path)

        enc = TurboEncoder(
            rate=1 / 3,
            terminate=True,
            constraint_length=4,
            device=device,
        )
        # Use 10 iterations as in the TensorFlow reference test
        dec = TurboDecoder(enc, num_iter=10, hard_out=True, device=device)

        # Apply the same LLR scaling as the TensorFlow reference test
        # The TF test uses: dec(-4.*yref/no)
        y = torch.tensor(-4.0 * yref / no, dtype=torch.float32, device=device)
        u_hat = dec(y)

        assert np.array_equal(u_hat.cpu().numpy(), uref)

    @pytest.mark.parametrize(
        "dt,precision",
        [(torch.float32, "single"), (torch.float64, "double")],
    )
    def test_dtype_flexible(self, device, dt, precision):
        """Test that decoder supports variable dtypes."""
        bs = 10
        k = 32

        source = BinarySource(device=device)

        enc = TurboEncoder(rate=0.5, constraint_length=4, device=device)
        u = source([bs, k])
        c = enc(u)
        y = 20.0 * (2.0 * c - 1.0)

        dec = TurboDecoder(
            enc, num_iter=4, hard_out=True, precision=precision, device=device
        )
        y_dt = y.to(dt)
        u_hat = dec(y_dt)

        # Check correct output dtype
        assert u_hat.dtype == dt

    @pytest.mark.parametrize("t", [False, True])
    def test_torch_compile(self, device, t):
        """Test that torch.compile works as expected."""
        bs = 10
        k = 100

        source = BinarySource(device=device)

        enc = TurboEncoder(
            rate=0.5, constraint_length=5, terminate=t, device=device
        )
        dec = TurboDecoder(enc, num_iter=4, hard_out=True, device=device)

        compiled_dec = torch.compile(dec)

        u = source([bs, k])
        c = enc(u)
        y = 20.0 * (2.0 * c - 1.0)

        x = compiled_dec(y)

        # Execute twice
        x2 = compiled_dec(y)
        assert torch.equal(x, x2)

    @pytest.mark.parametrize("k", [40, 100, 200])
    def test_dynamic_shapes(self, device, k):
        """Test that decoder works with different input shapes."""
        source = BinarySource(device=device)

        enc = TurboEncoder(rate=0.5, constraint_length=4, terminate=True, device=device)
        dec = TurboDecoder(enc, num_iter=4, hard_out=True, device=device)

        u = source([10, k])
        c = enc(u)
        y = 20.0 * (2.0 * c - 1.0)
        u_hat = dec(y)

        assert u_hat.shape[-1] == k
        # With high SNR, should recover input
        assert torch.equal(u, u_hat)

    def test_decoder_without_encoder(self, device):
        """Test that decoder can be constructed without an encoder object."""
        k = 100

        source = BinarySource(device=device)

        enc = TurboEncoder(
            rate=1 / 3, constraint_length=4, terminate=True, device=device
        )
        # Decoder from encoder
        dec1 = TurboDecoder(enc, num_iter=4, hard_out=True, device=device)
        # Decoder from explicit params
        dec2 = TurboDecoder(
            rate=1 / 3,
            constraint_length=4,
            terminate=True,
            interleaver="3GPP",
            num_iter=4,
            hard_out=True,
            device=device,
        )

        u = source([10, k])
        c = enc(u)
        y = 20.0 * (2.0 * c - 1.0)

        u_hat1 = dec1(y)
        u_hat2 = dec2(y)

        assert torch.equal(u_hat1, u_hat2)
        assert torch.equal(u, u_hat1)

    def test_soft_output(self, device):
        """Test soft (LLR) output mode."""
        bs = 10
        k = 100

        source = BinarySource(device=device)
        enc = TurboEncoder(
            rate=1 / 3, constraint_length=4, terminate=True, device=device
        )
        dec = TurboDecoder(enc, num_iter=4, hard_out=False, device=device)

        u = source([bs, k])
        c = enc(u)
        y = 20.0 * (2.0 * c - 1.0)

        llr_out = dec(y)

        # LLR should be continuous values
        assert llr_out.shape[-1] == k

        # Sign of LLR should match hard decision
        u_hat = (llr_out > 0).float()
        assert torch.equal(u, u_hat)

    @pytest.mark.parametrize("algo", ["map", "log", "maxlog"])
    def test_algorithms(self, device, algo):
        """Test different BCJR algorithms."""
        bs = 10
        k = 50

        source = BinarySource(device=device)
        enc = TurboEncoder(
            rate=1 / 3, constraint_length=4, terminate=True, device=device
        )

        u = source([bs, k])
        c = enc(u)
        y = 20.0 * (2.0 * c - 1.0)

        dec = TurboDecoder(
            enc, num_iter=4, hard_out=True, algorithm=algo, device=device
        )
        u_hat = dec(y)

        # All algorithms should recover input with perfect channel
        assert torch.equal(u, u_hat)

    def test_docstring_example(self):
        """Verify the docstring example works correctly."""
        encoder = TurboEncoder(rate=1 / 3, constraint_length=4, terminate=True)
        decoder = TurboDecoder(encoder, num_iter=6)

        u = torch.randint(0, 2, (10, 40), dtype=torch.float32)
        c = encoder(u)

        # Simulate BPSK with AWGN
        x = 2.0 * c - 1.0
        y = x + 0.5 * torch.randn_like(x)
        llr = 2.0 * y / 0.25

        u_hat = decoder(llr)
        assert u_hat.shape == torch.Size([10, 40])

