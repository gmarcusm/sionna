#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH Channel Estimation for the 5G NR module of Sionna PHY."""

from typing import Optional, Tuple, Union
import torch

from sionna.phy.ofdm import LSChannelEstimator
from sionna.phy.utils import expand_to_rank, split_dim


__all__ = ["PUSCHLSChannelEstimator"]


class PUSCHLSChannelEstimator(LSChannelEstimator):
    r"""Least-squares (LS) channel estimation for NR PUSCH transmissions.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated across the entire resource grid using
    a specified interpolation function.

    The implementation is similar to that of
    :class:`~sionna.phy.ofdm.LSChannelEstimator`.
    However, it additionally takes into account the separation of streams
    in the same CDM group as defined in
    :class:`~sionna.phy.nr.PUSCHDMRSConfig`. This is done through
    frequency and time averaging of adjacent LS channel estimates.

    :param resource_grid: ResourceGrid to be used.
    :param dmrs_length: Length of DMRS symbols. Must be 1 or 2.
        See :class:`~sionna.phy.nr.PUSCHDMRSConfig`.
    :param dmrs_additional_position: Number of additional DMRS symbols.
        Must be 0, 1, 2, or 3.
        See :class:`~sionna.phy.nr.PUSCHDMRSConfig`.
    :param num_cdm_groups_without_data: Number of CDM groups masked for
        data transmissions. Must be 1, 2, or 3.
        See :class:`~sionna.phy.nr.PUSCHDMRSConfig`.
    :param interpolation_type: The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are
        :class:`~sionna.phy.ofdm.NearestNeighborInterpolator` (``"nn"``),
        :class:`~sionna.phy.ofdm.LinearInterpolator` without (``"lin"``)
        or with averaging across OFDM symbols (``"lin_time_avg"``).
        Defaults to ``"nn"``.
    :param interpolator: Interpolator such as
        :class:`~sionna.phy.ofdm.LMMSEInterpolator`, or `None`.
        In the latter case, the interpolator specified by
        ``interpolation_type`` is used. Otherwise, the ``interpolator``
        is used and ``interpolation_type`` is ignored.
        Defaults to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Observed resource grid.

    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :output h_ls: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.

    :output err_var: Same shape as ``h_ls``, `torch.float`.
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHLSChannelEstimator

        config = PUSCHConfig()
        transmitter = PUSCHTransmitter(config)
        estimator = PUSCHLSChannelEstimator(
            transmitter.resource_grid,
            dmrs_length=1,
            dmrs_additional_position=0,
            num_cdm_groups_without_data=1
        )
    """

    def __init__(
        self,
        resource_grid,
        dmrs_length: int,
        dmrs_additional_position: int,
        num_cdm_groups_without_data: int,
        interpolation_type: str = "nn",
        interpolator=None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            resource_grid,
            interpolation_type,
            interpolator,
            precision=precision,
            device=device,
            **kwargs,
        )

        self._dmrs_length = dmrs_length
        self._dmrs_additional_position = dmrs_additional_position
        self._num_cdm_groups_without_data = num_cdm_groups_without_data

        # Number of DMRS OFDM symbols
        self._num_dmrs_syms = self._dmrs_length * (self._dmrs_additional_position + 1)

        # Number of pilot symbols per DMRS OFDM symbol
        self._num_pilots_per_dmrs_sym = int(
            self._pilot_pattern.pilots.shape[-1] / self._num_dmrs_syms)

    def estimate_at_pilot_locations(
        self,
        y_pilots: torch.Tensor,
        no: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LS channel estimation at pilot locations with CDM processing.

        :param y_pilots: Observed signals for the pilot-carrying resource
            elements.
            Shape: [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols]
        :param no: Variance of the AWGN.
        """
        # Compute LS channel estimates
        # Use safe division to avoid inf/nan from division by zero
        # (torch.where evaluates both branches before selecting)
        pilots = self._pilot_pattern.pilots
        pilot_mask = pilots.abs() > 0
        safe_pilots = torch.where(pilot_mask, pilots, torch.ones_like(pilots))
        h_ls = torch.where(
            pilot_mask,
            y_pilots / safe_pilots,
            torch.zeros_like(y_pilots))
        h_ls_shape = h_ls.shape

        # Compute error variance and broadcast to same shape as h_ls
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, device=self.device, dtype=self.dtype)
        no = expand_to_rank(no, h_ls.dim(), -1)
        pilots_expanded = expand_to_rank(pilots, h_ls.dim(), 0)
        pilot_mask_expanded = pilots_expanded.abs() > 0
        safe_pilots_expanded = torch.where(
            pilot_mask_expanded, pilots_expanded, torch.ones_like(pilots_expanded))

        # Compute error variance with safe division
        err_var = torch.where(
            pilot_mask_expanded,
            no / safe_pilots_expanded.abs().pow(2),
            torch.zeros_like(no))

        # Deal with CDM through time and frequency averaging
        h_hat = h_ls

        # (Optional) Time-averaging across adjacent DMRS OFDM symbols
        if self._dmrs_length == 2:
            # Reshape last dim to [num_dmrs_syms, num_pilots_per_dmrs_sym]
            h_hat = split_dim(h_hat, [self._num_dmrs_syms,
                                      self._num_pilots_per_dmrs_sym], 5)

            # Average adjacent DMRS symbols in time domain
            h_hat_even = h_hat[..., 0::2, :]
            h_hat_odd = h_hat[..., 1::2, :]
            h_hat_avg = (h_hat_even + h_hat_odd) / 2.0
            h_hat = h_hat_avg.repeat_interleave(2, dim=-2)
            h_hat = h_hat.reshape(h_ls_shape)

            # Error variance reduced by factor of 2
            err_var = err_var / 2.0

        # Frequency-averaging between adjacent channel estimates
        n = 2 * self._num_cdm_groups_without_data
        k = int(h_hat.shape[-1] / n)

        # Reshape last dimension to [k, n]
        h_hat = split_dim(h_hat, [k, n], 5)
        cond = h_hat.abs() > 0  # Mask for irrelevant estimates
        h_hat_sum = h_hat.sum(dim=-1, keepdim=True) / 2.0
        h_hat = h_hat_sum.repeat_interleave(n, dim=-1)
        h_hat = torch.where(cond, h_hat, torch.zeros_like(h_hat))
        h_hat = h_hat.reshape(h_ls_shape)

        # Error variance reduced by factor of 2
        err_var = err_var / 2.0

        return h_hat, err_var

