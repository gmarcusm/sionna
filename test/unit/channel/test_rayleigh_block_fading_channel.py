#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the RayleighBlockFading channel model"""

import torch

from sionna.phy.channel import (
    RayleighBlockFading,
    TimeChannel,
    GenerateTimeChannel,
    ApplyTimeChannel,
)


class TestRayleighBlockFading:
    """Tests for the RayleighBlockFading class"""

    def test_output_shape(self, device, precision):
        """Verify output shapes for path coefficients and delays"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 3
        num_tx_ant = 2
        batch_size = 16
        num_time_steps = 14

        channel = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision=precision,
            device=device,
        )
        h, tau = channel(batch_size, num_time_steps)

        expected_h_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 1, num_time_steps]
        )
        expected_tau_shape = torch.Size([batch_size, num_rx, num_tx, 1])

        assert h.shape == expected_h_shape
        assert tau.shape == expected_tau_shape

    def test_output_dtype(self, device, precision):
        """Verify output dtypes match precision setting"""
        channel = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=2,
            num_tx=1,
            num_tx_ant=4,
            precision=precision,
            device=device,
        )
        h, tau = channel(batch_size=32, num_time_steps=14)

        expected_cdtype = torch.complex64 if precision == "single" else torch.complex128
        expected_dtype = torch.float32 if precision == "single" else torch.float64

        assert h.dtype == expected_cdtype
        assert tau.dtype == expected_dtype

    def test_output_device(self, device):
        """Verify outputs are on the correct device"""
        channel = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        h, tau = channel(batch_size=32, num_time_steps=14)

        assert h.device == torch.device(device)
        assert tau.device == torch.device(device)

    def test_zero_delays(self, device):
        """Verify that all path delays are zero"""
        channel = RayleighBlockFading(
            num_rx=2, num_rx_ant=4, num_tx=3, num_tx_ant=2, device=device
        )
        _, tau = channel(batch_size=32, num_time_steps=14)

        assert torch.all(tau == 0.0)

    def test_block_fading_constant_over_time(self, device, precision):
        """Verify that channel coefficients are constant over time steps (block fading)"""
        channel = RayleighBlockFading(
            num_rx=2,
            num_rx_ant=4,
            num_tx=3,
            num_tx_ant=2,
            precision=precision,
            device=device,
        )
        h, _ = channel(batch_size=16, num_time_steps=100)

        # All time steps should have the same channel coefficient
        # Compare each time step to the first one
        h_first = h[..., 0:1]
        assert torch.allclose(h, h_first.expand_as(h))

    def test_channel_variance(self, device):
        """Verify that channel coefficients have unit variance (Rayleigh)"""
        channel = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            precision="double",
            device=device,
        )

        # Generate many samples to estimate variance
        batch_size = 100000
        h, _ = channel(batch_size=batch_size, num_time_steps=1)

        # h should be complex Gaussian with variance 1 (0.5 per real/imag part)
        # |h|^2 should have mean 1
        power = torch.abs(h.flatten()) ** 2
        mean_power = power.mean().item()

        assert abs(mean_power - 1.0) < 0.05

    def test_channel_real_imag_variance(self, device):
        """Verify that real and imaginary parts each have variance 0.5"""
        channel = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            precision="double",
            device=device,
        )

        batch_size = 100000
        h, _ = channel(batch_size=batch_size, num_time_steps=1)

        h_flat = h.flatten()
        real_var = h_flat.real.var().item()
        imag_var = h_flat.imag.var().item()

        assert abs(real_var - 0.5) < 0.05
        assert abs(imag_var - 0.5) < 0.05

    def test_channel_mean(self, device):
        """Verify that channel coefficients have zero mean"""
        channel = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            precision="double",
            device=device,
        )

        batch_size = 100000
        h, _ = channel(batch_size=batch_size, num_time_steps=1)

        h_flat = h.flatten()
        real_mean = h_flat.real.mean().item()
        imag_mean = h_flat.imag.mean().item()

        assert abs(real_mean) < 0.05
        assert abs(imag_mean) < 0.05

    def test_properties(self, device):
        """Verify that all properties return correct values"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 3
        num_tx_ant = 5

        channel = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        assert channel.num_rx == num_rx
        assert channel.num_rx_ant == num_rx_ant
        assert channel.num_tx == num_tx
        assert channel.num_tx_ant == num_tx_ant

    def test_sampling_frequency_ignored(self, device):
        """Verify that sampling_frequency parameter is accepted but ignored"""
        channel = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        # Should not raise an error when sampling_frequency is provided
        h1, tau1 = channel(batch_size=32, num_time_steps=14, sampling_frequency=15e3)
        h2, tau2 = channel(batch_size=32, num_time_steps=14, sampling_frequency=None)

        # Shapes should be the same
        assert h1.shape == h2.shape
        assert tau1.shape == tau2.shape

    def test_single_antenna_single_link(self, device):
        """Verify correct behavior for simplest case: SISO"""
        channel = RayleighBlockFading(
            num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1, device=device
        )
        h, tau = channel(batch_size=8, num_time_steps=10)

        assert h.shape == torch.Size([8, 1, 1, 1, 1, 1, 10])
        assert tau.shape == torch.Size([8, 1, 1, 1])

    def test_mimo_configuration(self, device):
        """Verify correct behavior for MIMO configuration"""
        channel = RayleighBlockFading(
            num_rx=4, num_rx_ant=8, num_tx=2, num_tx_ant=16, device=device
        )
        h, tau = channel(batch_size=64, num_time_steps=28)

        assert h.shape == torch.Size([64, 4, 8, 2, 16, 1, 28])
        assert tau.shape == torch.Size([64, 4, 2, 1])

    def test_docstring_example(self, device):
        """Test the example from the docstring"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        h, tau = channel_model(batch_size=32, num_time_steps=14)
        assert h.shape == torch.Size([32, 1, 2, 1, 4, 1, 14])
        assert tau.shape == torch.Size([32, 1, 1, 1])


class TestRayleighBlockFadingWithOFDMChannel:
    """Integration tests for RayleighBlockFading with OFDMChannel"""

    def test_with_ofdm_channel(self, device):
        """Verify RayleighBlockFading works with OFDMChannel"""
        from sionna.phy.channel import OFDMChannel

        # Create a simple resource grid-like object
        class SimpleResourceGrid:
            num_ofdm_symbols = 14
            fft_size = 64
            subcarrier_spacing = 15e3
            cyclic_prefix_length = 4

            @property
            def ofdm_symbol_duration(self):
                return (1 + self.cyclic_prefix_length / self.fft_size) / self.subcarrier_spacing

        rg = SimpleResourceGrid()
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        ofdm_channel = OFDMChannel(channel_model, rg, return_channel=True, device=device)

        x = torch.randn(
            32, 1, 4, 14, 64, dtype=torch.complex64, device=device
        )
        y, h_freq = ofdm_channel(x)

        assert y.shape == torch.Size([32, 1, 2, 14, 64])
        assert h_freq.shape == torch.Size([32, 1, 2, 1, 4, 14, 64])


class TestRayleighBlockFadingCompiled:
    """Tests for RayleighBlockFading with torch.compile"""

    def test_compiled(self, device):
        """Verify RayleighBlockFading works with torch.compile"""
        channel = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        @torch.compile
        def func(batch_size, num_time_steps):
            return channel(batch_size, num_time_steps)

        h, tau = func(32, 14)
        assert h.shape == torch.Size([32, 1, 2, 1, 4, 1, 14])
        assert tau.shape == torch.Size([32, 1, 1, 1])


class TestRayleighBlockFadingWithTimeChannel:
    """Integration tests for RayleighBlockFading with TimeChannel"""

    def test_time_channel_output_shape(self, device, precision):
        """Verify TimeChannel output shapes when using RayleighBlockFading"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        batch_size = 32
        num_time_samples = 100
        l_min = -6
        l_max = 20
        l_tot = l_max - l_min + 1

        channel_model = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision=precision,
            device=device,
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            return_channel=True,
            precision=precision,
            device=device,
        )

        x = torch.randn(
            batch_size, num_tx, num_tx_ant, num_time_samples,
            dtype=torch.complex64 if precision == "single" else torch.complex128,
            device=device,
        )
        y, h_time = time_channel(x)

        expected_y_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_time_samples + l_max - l_min]
        )
        expected_h_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 
             num_time_samples + l_max - l_min, l_tot]
        )

        assert y.shape == expected_y_shape
        assert h_time.shape == expected_h_shape

    def test_time_channel_output_dtype(self, device, precision):
        """Verify TimeChannel output dtypes match precision"""
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=2,
            num_tx=1,
            num_tx_ant=4,
            precision=precision,
            device=device,
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            return_channel=True,
            precision=precision,
            device=device,
        )

        cdtype = torch.complex64 if precision == "single" else torch.complex128
        x = torch.randn(32, 1, 4, 100, dtype=cdtype, device=device)
        y, h_time = time_channel(x)

        assert y.dtype == cdtype
        assert h_time.dtype == cdtype

    def test_time_channel_mimo(self, device):
        """Verify TimeChannel works with MIMO configurations"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 2
        num_tx_ant = 8
        batch_size = 16
        num_time_samples = 50
        l_min = -3
        l_max = 10

        channel_model = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            return_channel=True,
            device=device,
        )

        x = torch.randn(
            batch_size, num_tx, num_tx_ant, num_time_samples,
            dtype=torch.complex64, device=device
        )
        y, h_time = time_channel(x)

        # Output should sum over all TX and TX antennas
        expected_y_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_time_samples + l_max - l_min]
        )
        assert y.shape == expected_y_shape

    def test_time_channel_no_return_channel(self, device):
        """Verify TimeChannel works when return_channel is False"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            return_channel=False,
            device=device,
        )

        x = torch.randn(32, 1, 4, 100, dtype=torch.complex64, device=device)
        y = time_channel(x)

        # Should return only output, not a tuple
        assert isinstance(y, torch.Tensor)
        assert y.shape == torch.Size([32, 1, 2, 126])

    def test_time_channel_with_noise(self, device):
        """Verify TimeChannel correctly adds noise"""
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            precision="double",
            device=device,
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=1000,
            l_min=0,
            l_max=0,  # Single tap channel for simplicity
            return_channel=True,
            precision="double",
            device=device,
        )

        # Use zero input to isolate noise
        x = torch.zeros(10000, 1, 1, 1000, dtype=torch.complex128, device=device)
        no = 0.5
        y, _ = time_channel(x, no=no)

        # Estimate noise variance
        noise_var = y.var().item()
        assert abs(noise_var - no) < 0.1

    def test_time_channel_docstring_example(self, device):
        """Test the TimeChannel docstring example"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            return_channel=True,
            device=device,
        )

        x = torch.randn(32, 1, 4, 100, dtype=torch.complex64, device=device)
        y, h_time = channel(x)
        assert y.shape == torch.Size([32, 1, 2, 126])
        assert h_time.shape == torch.Size([32, 1, 2, 1, 4, 126, 27])


class TestRayleighBlockFadingWithGenerateTimeChannel:
    """Integration tests for RayleighBlockFading with GenerateTimeChannel"""

    def test_generate_time_channel_output_shape(self, device, precision):
        """Verify GenerateTimeChannel output shape with RayleighBlockFading"""
        num_rx = 2
        num_rx_ant = 4
        num_tx = 3
        num_tx_ant = 2
        batch_size = 16
        num_time_samples = 100
        l_min = -6
        l_max = 20
        l_tot = l_max - l_min + 1

        channel_model = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            precision=precision,
            device=device,
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            precision=precision,
            device=device,
        )

        h_time = gen_channel(batch_size)

        expected_shape = torch.Size(
            [
                batch_size,
                num_rx,
                num_rx_ant,
                num_tx,
                num_tx_ant,
                num_time_samples + l_max - l_min,
                l_tot,
            ]
        )
        assert h_time.shape == expected_shape

    def test_generate_time_channel_normalization(self, device):
        """Verify channel normalization produces unit average energy per time step"""
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            precision="double",
            device=device,
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            normalize_channel=True,
            precision="double",
            device=device,
        )

        # Generate many realizations
        h_time = gen_channel(batch_size=10000)

        # Average energy per time step should be close to 1
        # Sum over taps (last dim), square, average over batch and time
        energy_per_time = (torch.abs(h_time) ** 2).sum(dim=-1).mean()
        assert abs(energy_per_time.item() - 1.0) < 0.1

    def test_generate_time_channel_docstring_example(self, device):
        """Test the GenerateTimeChannel docstring example"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )
        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            device=device,
        )
        h_time = gen_channel(batch_size=32)
        assert h_time.shape == torch.Size([32, 1, 2, 1, 4, 126, 27])


class TestRayleighBlockFadingWithApplyTimeChannel:
    """Integration tests for RayleighBlockFading with ApplyTimeChannel"""

    def test_apply_time_channel_output_shape(self, device):
        """Verify ApplyTimeChannel output shape when using RayleighBlockFading-generated h_time"""
        num_rx = 1
        num_rx_ant = 2
        num_tx = 1
        num_tx_ant = 4
        batch_size = 32
        num_time_samples = 100
        l_min = -6
        l_max = 20
        l_tot = l_max - l_min + 1

        channel_model = RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=num_time_samples,
            l_min=l_min,
            l_max=l_max,
            device=device,
        )

        apply_channel = ApplyTimeChannel(
            num_time_samples=num_time_samples,
            l_tot=l_tot,
            device=device,
        )

        h_time = gen_channel(batch_size)
        x = torch.randn(
            batch_size, num_tx, num_tx_ant, num_time_samples,
            dtype=torch.complex64, device=device
        )
        y = apply_channel(x, h_time)

        expected_y_shape = torch.Size(
            [batch_size, num_rx, num_rx_ant, num_time_samples + l_tot - 1]
        )
        assert y.shape == expected_y_shape

    def test_apply_time_channel_linear_operation(self, device):
        """Verify ApplyTimeChannel performs correct filtering for simple case"""
        # Use SISO with single tap (flat fading) for easy verification
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            precision="double",
            device=device,
        )

        gen_channel = GenerateTimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=10,
            l_min=0,
            l_max=0,  # Single tap
            precision="double",
            device=device,
        )

        apply_channel = ApplyTimeChannel(
            num_time_samples=10,
            l_tot=1,
            precision="double",
            device=device,
        )

        h_time = gen_channel(batch_size=8)
        x = torch.randn(8, 1, 1, 10, dtype=torch.complex128, device=device)
        y = apply_channel(x, h_time)

        # For single tap, y = h * x (element-wise, summed over tx)
        # h_time shape: [8, 1, 1, 1, 1, 10, 1]
        # x shape: [8, 1, 1, 10]
        h_flat = h_time.squeeze()  # [8, 10]
        x_flat = x.squeeze()  # [8, 10]
        y_expected = h_flat * x_flat
        y_actual = y.squeeze()  # [8, 10]

        assert torch.allclose(y_actual, y_expected, atol=1e-10)


class TestRayleighBlockFadingTimeChannelCompiled:
    """Tests for RayleighBlockFading with TimeChannel and torch.compile"""

    def test_time_channel_compiled(self, device):
        """Verify TimeChannel with RayleighBlockFading works with torch.compile"""
        channel_model = RayleighBlockFading(
            num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=4, device=device
        )

        time_channel = TimeChannel(
            channel_model,
            bandwidth=1e6,
            num_time_samples=100,
            l_min=-6,
            l_max=20,
            return_channel=True,
            device=device,
        )

        @torch.compile
        def func(x):
            return time_channel(x)

        x = torch.randn(32, 1, 4, 100, dtype=torch.complex64, device=device)
        y, h_time = func(x)
        assert y.shape == torch.Size([32, 1, 2, 126])
        assert h_time.shape == torch.Size([32, 1, 2, 1, 4, 126, 27])

