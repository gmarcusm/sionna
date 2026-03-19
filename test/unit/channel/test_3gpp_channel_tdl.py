#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the 3GPP TDL channel model"""

import gc
import numpy as np
import pytest
import torch
from scipy.stats import kstest, rayleigh, rice
from scipy.special import jv

from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import exp_corr_mat
from channel_test_utils import (
    TDL_POWERS,
    TDL_DELAYS,
    TDL_RICIAN_K,
)


def _cleanup_gpu_memory(device):
    """Clean up GPU memory if running on CUDA."""
    gc.collect()
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.empty_cache()


class TestTDL:
    """Tests for the 3GPP TDL channel model"""

    # Test parameters
    BATCH_SIZE = 10000
    CARRIER_FREQUENCY = 3.5e9  # Hz
    SAMPLING_FREQUENCY = 15e3  # Hz
    DELAY_SPREAD = 100e-9  # s
    NUM_TIME_STEPS = 100
    NUM_SINUSOIDS = 20
    SPEED = 150  # m/s
    MAX_DOPPLER = 2. * PI * SPEED / SPEED_OF_LIGHT * CARRIER_FREQUENCY
    LoS_AoA = np.pi / 4
    MAX_ERR = 5e-2

    @pytest.mark.parametrize("model", ['A', 'B', 'C', 'D', 'E', 'A30', 'B100', 'C300'])
    def test_pdp(self, model, device):
        """Test power delay profiles match expected values"""
        # Set delay spread based on model
        if model == 'A30':
            delay_spread = 30e-9
        elif model == 'B100':
            delay_spread = 100e-9
        elif model == 'C300':
            delay_spread = 300e-9
        else:
            delay_spread = self.DELAY_SPREAD

        # Create TDL model
        tdl = TDL(
            model=model,
            delay_spread=delay_spread,
            carrier_frequency=self.CARRIER_FREQUENCY,
            num_sinusoids=self.NUM_SINUSOIDS,
            los_angle_of_arrival=self.LoS_AoA,
            min_speed=self.SPEED,
            device=device,
        )
        h, tau = tdl(
            batch_size=self.BATCH_SIZE,
            num_time_steps=self.NUM_TIME_STEPS,
            sampling_frequency=self.SAMPLING_FREQUENCY,
        )
        h_np = h[:, 0, 0, 0, 0, :, :].cpu().numpy()
        tau_np = tau[:, 0, 0, :].cpu().numpy()

        # Check powers
        p = np.mean(np.square(np.abs(h_np[:, :, 0])), axis=0)
        ref_p = np.power(10.0, TDL_POWERS[model] / 10.0)
        ref_p = ref_p / np.sum(ref_p)
        max_err = np.max(np.abs(ref_p - p))
        assert max_err <= self.MAX_ERR, f"Power profile error for {model}: {max_err}"

        # Check delays
        if model in ('A30', 'B100', 'C300'):
            ref_tau = np.expand_dims(TDL_DELAYS[model], axis=0) * 1e-9  # ns to s
        else:
            tau_np = tau_np / self.DELAY_SPREAD
            ref_tau = np.expand_dims(TDL_DELAYS[model], axis=0)
        max_err = np.max(np.abs(ref_tau - tau_np))
        assert max_err <= self.MAX_ERR, f"Delay error for {model}: {max_err}"

    @pytest.mark.parametrize("model", ['A', 'B', 'C', 'D', 'E', 'A30', 'B100', 'C300'])
    def test_taps_powers_distributions(self, model, device):
        """Test the distribution of the taps powers"""
        # Set delay spread based on model
        if model == 'A30':
            delay_spread = 30e-9
        elif model == 'B100':
            delay_spread = 100e-9
        elif model == 'C300':
            delay_spread = 300e-9
        else:
            delay_spread = self.DELAY_SPREAD

        # Create TDL model
        tdl = TDL(
            model=model,
            delay_spread=delay_spread,
            carrier_frequency=self.CARRIER_FREQUENCY,
            num_sinusoids=self.NUM_SINUSOIDS,
            los_angle_of_arrival=self.LoS_AoA,
            min_speed=self.SPEED,
            device=device,
        )
        h, _ = tdl(
            batch_size=self.BATCH_SIZE,
            num_time_steps=self.NUM_TIME_STEPS,
            sampling_frequency=self.SAMPLING_FREQUENCY,
        )
        h_np = h[:, 0, 0, 0, 0, :, :].cpu().numpy()

        ref_powers = np.power(10.0, TDL_POWERS[model] / 10.0)
        ref_powers = ref_powers / np.sum(ref_powers)
        powers = np.abs(h_np)

        for i, p in enumerate(ref_powers):
            if i == 0 and (model == 'D' or model == 'E'):
                # First tap of LoS models follows Rice distribution
                K = np.power(10.0, TDL_RICIAN_K[model] / 10.0)
                P0 = ref_powers[0]
                s = np.sqrt(0.5 * P0 / (1 + K))
                b = np.sqrt(K * 2)
                D, _ = kstest(
                    powers[:, i, 0].flatten(),
                    rice.cdf,
                    args=(b, 0.0, s)
                )
            else:
                # NLoS taps follow Rayleigh distribution
                D, _ = kstest(
                    powers[:, i, 0].flatten(),
                    rayleigh.cdf,
                    args=(0.0, np.sqrt(0.5 * p))
                )
            assert D <= self.MAX_ERR, f"Distribution error for {model} tap {i}: {D}"

    def _corr(self, x, max_lags):
        """Compute autocorrelation"""
        num_lags = x.shape[-1] // 2
        c = np.zeros([max_lags], dtype=complex)
        for i in range(0, max_lags):
            c[i] = np.mean(np.conj(x[..., :num_lags]) * x[..., i:num_lags + i])
        return c

    def _auto_real(self, max_doppler, t, power):
        """Autocorrelation of real part (Eq. 8a)"""
        return 0.5 * power * jv(0, t * max_doppler)

    def _auto_complex(self, max_doppler, t, power):
        """Autocorrelation of complex signal (Eq. 8c)"""
        return power * jv(0, t * max_doppler)

    def _auto_complex_rice(self, max_doppler, K, theta_0, t):
        """Autocorrelation for Rice channel"""
        a = jv(0, t * max_doppler)
        b = K * np.cos(t * max_doppler * np.cos(theta_0))
        c = 1j * K * np.sin(t * max_doppler * np.cos(theta_0))
        return (a + b + c) / (1 + K)

    @pytest.mark.parametrize("model", ['A', 'B', 'C', 'D', 'E', 'A30', 'B100', 'C300'])
    def test_autocorrelation(self, model, device):
        """Test the temporal autocorrelation matches theoretical values"""
        # Set delay spread based on model
        if model == 'A30':
            delay_spread = 30e-9
        elif model == 'B100':
            delay_spread = 100e-9
        elif model == 'C300':
            delay_spread = 300e-9
        else:
            delay_spread = self.DELAY_SPREAD

        # Create TDL model
        tdl = TDL(
            model=model,
            delay_spread=delay_spread,
            carrier_frequency=self.CARRIER_FREQUENCY,
            num_sinusoids=self.NUM_SINUSOIDS,
            los_angle_of_arrival=self.LoS_AoA,
            min_speed=self.SPEED,
            device=device,
        )
        h, _ = tdl(
            batch_size=self.BATCH_SIZE,
            num_time_steps=self.NUM_TIME_STEPS,
            sampling_frequency=self.SAMPLING_FREQUENCY,
        )
        h_np = h[:, 0, 0, 0, 0, :, :].cpu().numpy()

        max_lag = self.NUM_TIME_STEPS // 2
        ref_powers = np.power(10.0, TDL_POWERS[model] / 10.0)
        ref_powers = ref_powers / np.sum(ref_powers)
        time = np.arange(max_lag) / self.SAMPLING_FREQUENCY

        for i, p in enumerate(ref_powers):
            if i == 0 and (model == 'D' or model == 'E'):
                # LoS model first tap
                h_tap = h_np[:, i, :]
                r = self._corr(h_tap, max_lag)

                K = np.power(10.0, TDL_RICIAN_K[model] / 10.0)
                ref_r = self._auto_complex_rice(
                    self.MAX_DOPPLER, K, self.LoS_AoA, time
                ) * p
                max_err = np.max(np.abs(r - ref_r))
                assert max_err <= self.MAX_ERR, \
                    f"Autocorrelation error for {model} tap {i}: {max_err}"
            else:
                # NLoS taps
                h_tap = h_np[:, i, :]
                r_real = self._corr(h_tap.real, max_lag)
                r_imag = self._corr(h_tap.imag, max_lag)
                r = self._corr(h_tap, max_lag)

                ref_r_real = self._auto_real(self.MAX_DOPPLER, time, p)
                max_err = np.max(np.abs(r_real - ref_r_real))
                assert max_err <= self.MAX_ERR, \
                    f"Real autocorrelation error for {model} tap {i}: {max_err}"

                max_err = np.max(np.abs(r_imag - ref_r_real))
                assert max_err <= self.MAX_ERR, \
                    f"Imag autocorrelation error for {model} tap {i}: {max_err}"

                ref_r = self._auto_complex(self.MAX_DOPPLER, time, p)
                max_err = np.max(np.abs(r - ref_r))
                assert max_err <= self.MAX_ERR, \
                    f"Complex autocorrelation error for {model} tap {i}: {max_err}"

    def test_spatial_correlation_separate_rx_tx(self, device):
        """Test spatial correlation with separate RX and TX correlation matrices"""
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant, device=device)
        tx_corr_mat = exp_corr_mat(0.5, num_tx_ant, device=device)

        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=0.0,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            rx_corr_mat=rx_corr_mat,
            tx_corr_mat=tx_corr_mat,
            device=device,
        )

        # Empirical estimation of the correlation matrices
        est_rx_cov = np.zeros([num_rx_ant, num_rx_ant], complex)
        est_tx_cov = np.zeros([num_tx_ant, num_tx_ant], complex)
        num_it = 100
        batch_size = 1000

        for i in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = h.permute(0, 1, 3, 5, 6, 2, 4).cpu().numpy()  # [..., rx ant, tx ant]
            h = h[:, 0, 0, 0, 0, :, :] / np.sqrt(tdl.mean_powers[0].cpu().numpy())

            # RX correlation
            h_ = np.expand_dims(h[:, :, 0], axis=-1)
            est_rx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_rx_cov_ = np.mean(est_rx_cov_, axis=0)
            est_rx_cov += est_rx_cov_

            # TX correlation
            h_ = np.expand_dims(h[:, 0, :], axis=-1)
            est_tx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_tx_cov_ = np.mean(est_tx_cov_, axis=0)
            est_tx_cov += est_tx_cov_

            # Periodic GPU memory cleanup
            if i % 50 == 49:
                _cleanup_gpu_memory(device)

        est_rx_cov /= num_it
        est_tx_cov /= num_it

        rx_corr_mat_np = rx_corr_mat.cpu().numpy()
        tx_corr_mat_np = tx_corr_mat.cpu().numpy()

        max_err = np.max(np.abs(est_rx_cov - rx_corr_mat_np))
        assert max_err <= self.MAX_ERR, f"Receiver correlation error: {max_err}"

        max_err = np.max(np.abs(est_tx_cov - tx_corr_mat_np))
        assert max_err <= self.MAX_ERR, f"Transmitter correlation error: {max_err}"

    def test_spatial_correlation_joint_rx_tx(self, device):
        """Test spatial correlation with joint filtering"""
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant // 2, device=device).cpu().numpy()
        pol_corr_mat = np.array([
            [1.0, 0.8, 0.0, 0.0],
            [0.8, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.8],
            [0.0, 0.0, 0.8, 1.0]
        ])
        tx_corr_mat = exp_corr_mat(0.5, num_tx_ant // 2, device=device).cpu().numpy()
        spatial_corr_mat = np.kron(pol_corr_mat, tx_corr_mat)
        spatial_corr_mat = np.kron(rx_corr_mat, spatial_corr_mat)

        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=0.0,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            spatial_corr_mat=spatial_corr_mat,
            device=device,
        )

        # Empirical estimation of the correlation matrices
        est_spatial_cov = np.zeros([num_tx_ant * num_rx_ant, num_tx_ant * num_rx_ant], complex)
        num_it = 100
        batch_size = 1000

        for i in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = h.permute(0, 1, 3, 5, 6, 2, 4).cpu().numpy()
            h = h[:, 0, 0, 0, 0, :, :] / np.sqrt(tdl.mean_powers[0].cpu().numpy())
            h = np.reshape(h, [batch_size, -1])

            h_ = np.expand_dims(h, axis=-1)
            est_spatial_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_spatial_cov_ = np.mean(est_spatial_cov_, axis=0)
            est_spatial_cov += est_spatial_cov_

            # Periodic GPU memory cleanup
            if i % 50 == 49:
                _cleanup_gpu_memory(device)

        est_spatial_cov /= num_it

        max_err = np.max(np.abs(est_spatial_cov - spatial_corr_mat))
        assert max_err <= self.MAX_ERR, f"Spatial correlation error: {max_err}"

    def test_no_spatial_correlation(self, device):
        """Test that no specified correlation leads to identity correlation"""
        num_rx_ant = 16
        num_tx_ant = 16

        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=0.0,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        # Empirical estimation of the correlation matrices
        est_spatial_cov = np.zeros([num_tx_ant * num_rx_ant, num_tx_ant * num_rx_ant], complex)
        num_it = 100
        batch_size = 1000

        for i in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = h.permute(0, 1, 3, 5, 6, 2, 4).cpu().numpy()
            h = h[:, 0, 0, 0, 0, :, :] / np.sqrt(tdl.mean_powers[0].cpu().numpy())
            h = np.reshape(h, [batch_size, -1])

            h_ = np.expand_dims(h, axis=-1)
            est_spatial_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_spatial_cov_ = np.mean(est_spatial_cov_, axis=0)
            est_spatial_cov += est_spatial_cov_

            # Periodic GPU memory cleanup
            if i % 50 == 49:
                _cleanup_gpu_memory(device)

        est_spatial_cov /= num_it

        spatial_corr_mat = np.eye(num_rx_ant * num_rx_ant)
        max_err = np.max(np.abs(est_spatial_cov - spatial_corr_mat))
        assert max_err <= self.MAX_ERR, f"Identity correlation error: {max_err}"

    def test_rx_corr_only(self, device):
        """Test with RX spatial correlation only"""
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant, device=device)
        tx_corr_mat_ref = np.eye(num_tx_ant)

        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=0.0,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            rx_corr_mat=rx_corr_mat,
            device=device,
        )

        # Empirical estimation of the correlation matrices
        est_rx_cov = np.zeros([num_rx_ant, num_rx_ant], complex)
        est_tx_cov = np.zeros([num_tx_ant, num_tx_ant], complex)
        num_it = 100
        batch_size = 1000

        for i in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = h.permute(0, 1, 3, 5, 6, 2, 4).cpu().numpy()
            h = h[:, 0, 0, 0, 0, :, :] / np.sqrt(tdl.mean_powers[0].cpu().numpy())

            # RX correlation
            h_ = np.expand_dims(h[:, :, 0], axis=-1)
            est_rx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_rx_cov_ = np.mean(est_rx_cov_, axis=0)
            est_rx_cov += est_rx_cov_

            # TX correlation
            h_ = np.expand_dims(h[:, 0, :], axis=-1)
            est_tx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_tx_cov_ = np.mean(est_tx_cov_, axis=0)
            est_tx_cov += est_tx_cov_

            # Periodic GPU memory cleanup
            if i % 50 == 49:
                _cleanup_gpu_memory(device)

        est_rx_cov /= num_it
        est_tx_cov /= num_it

        rx_corr_mat_np = rx_corr_mat.cpu().numpy()

        max_err = np.max(np.abs(est_rx_cov - rx_corr_mat_np))
        assert max_err <= self.MAX_ERR, f"Receiver correlation error: {max_err}"

        max_err = np.max(np.abs(est_tx_cov - tx_corr_mat_ref))
        assert max_err <= self.MAX_ERR, f"Transmitter correlation error: {max_err}"

    def test_tx_corr_only(self, device):
        """Test with TX spatial correlation only"""
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat_ref = np.eye(num_rx_ant)
        tx_corr_mat = exp_corr_mat(0.9, num_tx_ant, device=device)

        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=0.0,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            tx_corr_mat=tx_corr_mat,
            device=device,
        )

        # Empirical estimation of the correlation matrices
        est_rx_cov = np.zeros([num_rx_ant, num_rx_ant], complex)
        est_tx_cov = np.zeros([num_tx_ant, num_tx_ant], complex)
        num_it = 100
        batch_size = 1000

        for i in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = h.permute(0, 1, 3, 5, 6, 2, 4).cpu().numpy()
            h = h[:, 0, 0, 0, 0, :, :] / np.sqrt(tdl.mean_powers[0].cpu().numpy())

            # RX correlation
            h_ = np.expand_dims(h[:, :, 0], axis=-1)
            est_rx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_rx_cov_ = np.mean(est_rx_cov_, axis=0)
            est_rx_cov += est_rx_cov_

            # TX correlation
            h_ = np.expand_dims(h[:, 0, :], axis=-1)
            est_tx_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_tx_cov_ = np.mean(est_tx_cov_, axis=0)
            est_tx_cov += est_tx_cov_

            # Periodic GPU memory cleanup
            if i % 50 == 49:
                _cleanup_gpu_memory(device)

        est_rx_cov /= num_it
        est_tx_cov /= num_it

        tx_corr_mat_np = tx_corr_mat.cpu().numpy()

        max_err = np.max(np.abs(est_rx_cov - rx_corr_mat_ref))
        assert max_err <= self.MAX_ERR, f"Receiver correlation error: {max_err}"

        max_err = np.max(np.abs(est_tx_cov - tx_corr_mat_np))
        assert max_err <= self.MAX_ERR, f"Transmitter correlation error: {max_err}"

    def test_spatial_correlation_all_three_inputs(self, device):
        """Test that spatial_corr_mat takes priority over rx/tx_corr_mat"""
        num_rx_ant = 16
        num_tx_ant = 16
        rx_corr_mat = exp_corr_mat(0.9, num_rx_ant // 2, device=device).cpu().numpy()
        pol_corr_mat = np.array([
            [1.0, 0.8, 0.0, 0.0],
            [0.8, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.8],
            [0.0, 0.0, 0.8, 1.0]
        ])
        tx_corr_mat = exp_corr_mat(0.5, num_tx_ant // 2, device=device).cpu().numpy()
        spatial_corr_mat = np.kron(pol_corr_mat, tx_corr_mat)
        spatial_corr_mat = np.kron(rx_corr_mat, spatial_corr_mat)

        # Pass identity matrices for rx and tx - they should be ignored
        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=0.0,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            spatial_corr_mat=spatial_corr_mat,
            rx_corr_mat=np.eye(num_rx_ant),
            tx_corr_mat=np.eye(num_tx_ant),
            device=device,
        )

        # Empirical estimation of the correlation matrices
        est_spatial_cov = np.zeros([num_tx_ant * num_rx_ant, num_tx_ant * num_rx_ant], complex)
        num_it = 100
        batch_size = 1000

        for i in range(num_it):
            h, _ = tdl(batch_size, 1, 1)

            h = h.permute(0, 1, 3, 5, 6, 2, 4).cpu().numpy()
            h = h[:, 0, 0, 0, 0, :, :] / np.sqrt(tdl.mean_powers[0].cpu().numpy())
            h = np.reshape(h, [batch_size, -1])

            h_ = np.expand_dims(h, axis=-1)
            est_spatial_cov_ = np.matmul(h_, np.conj(np.transpose(h_, [0, 2, 1])))
            est_spatial_cov_ = np.mean(est_spatial_cov_, axis=0)
            est_spatial_cov += est_spatial_cov_

            # Periodic GPU memory cleanup
            if i % 50 == 49:
                _cleanup_gpu_memory(device)

        est_spatial_cov /= num_it

        max_err = np.max(np.abs(est_spatial_cov - spatial_corr_mat))
        assert max_err <= self.MAX_ERR, f"Spatial correlation priority error: {max_err}"

    def test_output_shapes(self, device):
        """Test that output shapes are correct"""
        batch_size = 32
        num_time_steps = 64
        num_rx_ant = 4
        num_tx_ant = 2

        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            device=device,
        )

        h, tau = tdl(batch_size, num_time_steps, 15e3)

        assert h.shape == (batch_size, 1, num_rx_ant, 1, num_tx_ant, tdl.num_clusters, num_time_steps)
        assert tau.shape == (batch_size, 1, 1, tdl.num_clusters)

    def test_output_dtype(self, device, precision):
        """Test that output dtypes match configured precision"""
        from sionna.phy import dtypes as sionna_dtypes

        cdtype = sionna_dtypes[precision]["torch"]["cdtype"]
        dtype = sionna_dtypes[precision]["torch"]["dtype"]

        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        h, tau = tdl(16, 32, 15e3)

        assert h.dtype == cdtype
        assert tau.dtype == dtype
        assert h.device == torch.device(device)
        assert tau.device == torch.device(device)

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        tdl = TDL(
            model="A",
            delay_spread=300e-9,
            carrier_frequency=3.5e9,
            min_speed=0.0,
            max_speed=3.0,
            device=device,
        )

        h, tau = tdl(batch_size=32, num_time_steps=14, sampling_frequency=15e3)

        assert h.shape[0] == 32
        assert h.shape[-1] == 14
        assert h.is_complex()

    def test_los_model_properties(self, device):
        """Test LoS model specific properties"""
        tdl_los = TDL(
            model="D",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            device=device,
        )

        assert tdl_los.los is True
        assert tdl_los.k_factor > 0
        assert tdl_los.mean_power_los > 0

    def test_nlos_model_properties(self, device):
        """Test NLoS model specific properties"""
        tdl_nlos = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            device=device,
        )

        assert tdl_nlos.los is False
        # K-factor and mean_power_los should raise assertion for NLoS
        with pytest.raises(AssertionError):
            _ = tdl_nlos.k_factor
        with pytest.raises(AssertionError):
            _ = tdl_nlos.mean_power_los

    def test_delay_spread_setter(self, device):
        """Test that delay spread can be modified for scalable models"""
        tdl = TDL(
            model="A",
            delay_spread=100e-9,
            carrier_frequency=3.5e9,
            device=device,
        )

        original_delays = tdl.delays.clone()
        tdl.delay_spread = 200e-9

        # Delays should change proportionally
        assert torch.allclose(tdl.delays, original_delays * 2, rtol=1e-5)

    def test_mean_powers_normalized(self, device):
        """Test that mean powers sum to 1 (normalized PDP)"""
        for model in ['A', 'B', 'C', 'D', 'E']:
            tdl = TDL(
                model=model,
                delay_spread=100e-9,
                carrier_frequency=3.5e9,
                device=device,
            )

            total_power = torch.sum(tdl.mean_powers)
            assert torch.isclose(total_power, torch.tensor(1.0, device=device), rtol=1e-5), \
                f"Model {model} powers don't sum to 1: {total_power}"
