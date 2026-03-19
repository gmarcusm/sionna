#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR38.901 rural macrocell (RMa) channel scenario"""

from typing import Optional

import torch

from sionna.phy import SPEED_OF_LIGHT, PI
from .system_level_scenario import SystemLevelScenario
from .antenna import PanelArray

__all__ = ["RMaScenario"]


class RMaScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 rural macrocell (RMa) channel model scenario.

    :param carrier_frequency: Carrier frequency [Hz]
    :param ut_array: Panel array used by the UTs. All UTs share the same
        antenna array configuration.
    :param bs_array: Panel array used by the BSs. All BSs share the same
        antenna array configuration.
    :param direction: Link direction. Either ``"uplink"`` or ``"downlink"``.
    :param enable_pathloss: If `True`, apply pathloss. Otherwise don't.
        Defaults to `True`.
    :param enable_shadow_fading: If `True`, apply shadow fading. Otherwise
        don't. Defaults to `True`.
    :param average_street_width: Average street width [m]. Defaults to 20.0.
    :param average_building_height: Average building height [m]. Defaults to
        5.0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    .. rubric:: Examples

    >>> from sionna.phy.channel.tr38901 import PanelArray, RMaScenario
    >>> # Configure antenna arrays
    >>> ut_array = PanelArray(num_rows_per_panel=1,
    ...                       num_cols_per_panel=1,
    ...                       polarization="single",
    ...                       polarization_type="V",
    ...                       antenna_pattern="omni",
    ...                       carrier_frequency=3.5e9)
    >>> bs_array = PanelArray(num_rows_per_panel=4,
    ...                       num_cols_per_panel=4,
    ...                       polarization="dual",
    ...                       polarization_type="cross",
    ...                       antenna_pattern="38.901",
    ...                       carrier_frequency=3.5e9)
    >>> scenario = RMaScenario(carrier_frequency=3.5e9,
    ...                        ut_array=ut_array,
    ...                        bs_array=bs_array,
    ...                        direction="downlink")
    """

    def __init__(
        self,
        carrier_frequency: float,
        ut_array: PanelArray,
        bs_array: PanelArray,
        direction: str,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        average_street_width: float = 20.0,
        average_building_height: float = 5.0,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        # Only the low-loss O2I model is available for RMa.
        super().__init__(
            carrier_frequency,
            "low",
            ut_array,
            bs_array,
            direction,
            enable_pathloss,
            enable_shadow_fading,
            precision=precision,
            device=device,
        )

        # Average street width [m]
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_average_street_width",
            torch.tensor(average_street_width, dtype=self.dtype, device=self.device),
        )

        # Average building height [m]
        self.register_buffer(
            "_average_building_height",
            torch.tensor(average_building_height, dtype=self.dtype, device=self.device),
        )

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc: torch.Tensor) -> torch.Tensor:
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation.

        :param fc: Carrier frequency [GHz]

        :output fc_clipped: `float`.
            Clipped carrier frequency, that should be used for LSP computation.
        """
        return fc

    @property
    def min_2d_in(self) -> torch.Tensor:
        """Minimum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def max_2d_in(self) -> torch.Tensor:
        """Maximum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(10.0, dtype=self.dtype, device=self.device)

    @property
    def average_street_width(self) -> torch.Tensor:
        """Average street width [m]"""
        return self._average_street_width

    @property
    def average_building_height(self) -> torch.Tensor:
        """Average building height [m]"""
        return self._average_building_height

    @property
    def los_probability(self) -> torch.Tensor:
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs.

        Computed following section 7.4.2 of TR 38.901.

        Shape [batch size, num_bs, num_ut]
        """
        distance_2d_out = self._distance_2d_out
        los_probability = torch.exp(-(distance_2d_out - 10.0) / 1000.0)
        los_probability = torch.where(
            distance_2d_out < 10.0,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            los_probability,
        )
        return los_probability

    @property
    def rays_per_cluster(self) -> int:
        """Number of rays per cluster"""
        return 20

    @property
    def los_parameter_filepath(self) -> str:
        """Path of the configuration file for LoS scenario"""
        return "RMa_LoS.json"

    @property
    def nlos_parameter_filepath(self) -> str:
        """Path of the configuration file for NLoS scenario"""
        return "RMa_NLoS.json"

    @property
    def o2i_parameter_filepath(self) -> str:
        """Path of the configuration file for indoor scenario"""
        return "RMa_O2I.json"

    #########################
    # Utility methods
    #########################

    def _compute_lsp_log_mean_std(self) -> None:
        r"""Computes the mean and standard deviations of LSPs in log-domain"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        h_bs = self.h_bs
        h_bs = h_bs.unsqueeze(2)  # For broadcasting
        h_ut = self.h_ut
        h_ut = h_ut.unsqueeze(1)  # For broadcasting

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF. Has zero-mean.
        log_mean_sf = torch.zeros(
            batch_size, num_bs, num_ut, dtype=self.dtype, device=self.device
        )
        # K. Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK") / 10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD mean is of the form max(-1, A*d2D/1000 - 0.01*(hUT-1.5) + B)
        log_mean_zsd = (
            self.get_param("muZSDa") * (distance_2d / 1000.0)
            - 0.01 * (h_ut - 1.5)
            + self.get_param("muZSDb")
        )
        log_mean_zsd = torch.maximum(
            torch.tensor(-1.0, dtype=self.dtype, device=self.device), log_mean_zsd
        )

        lsp_log_mean = torch.stack(
            [
                log_mean_ds,
                log_mean_asd,
                log_mean_asa,
                log_mean_sf,
                log_mean_k,
                log_mean_zsa,
                log_mean_zsd,
            ],
            dim=3,
        )

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        # O2I and NLoS cases just require the use of a predefined value
        log_std_sf_o2i_nlos = self.get_param("sigmaSF") / 10.0
        # For LoS, two possible scenarios depending on the 2D location of the user
        distance_breakpoint = (
            2.0 * PI * h_bs * h_ut * self.carrier_frequency / SPEED_OF_LIGHT
        )
        log_std_sf_los = torch.where(
            distance_2d < distance_breakpoint,
            self.get_param("sigmaSF1") / 10.0,
            self.get_param("sigmaSF2") / 10.0,
        )
        # Use the correct SF STD according to the user scenario: NLoS/O2I, or LoS
        log_std_sf = torch.where(self.los, log_std_sf_los, log_std_sf_o2i_nlos)
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK") / 10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")

        lsp_log_std = torch.stack(
            [
                log_std_ds,
                log_std_asd,
                log_std_asa,
                log_std_sf,
                log_std_k,
                log_std_zsa,
                log_std_zsd,
            ],
            dim=3,
        )

        self._update_attr("_lsp_log_mean", lsp_log_mean)
        self._update_attr("_lsp_log_std", lsp_log_std)

        # ZOD offset
        zod_offset = torch.atan((35.0 - 3.5) / distance_2d) - torch.atan(
            (35.0 - 1.5) / distance_2d
        )
        zod_offset = torch.where(
            self.los,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            zod_offset,
        )
        self._update_attr("_zod_offset", zod_offset)

    def _compute_pathloss_basic(self) -> None:
        r"""Computes the basic component of the pathloss [dB]"""

        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency / 1e9  # Carrier frequency (GHz)
        h_bs = self.h_bs
        h_bs = h_bs.unsqueeze(2)  # For broadcasting
        h_ut = self.h_ut
        h_ut = h_ut.unsqueeze(1)  # For broadcasting
        average_building_height = self.average_building_height

        # Break point distance
        # For this computation, the carrier frequency needs to be in Hz
        distance_breakpoint = (
            2.0 * PI * h_bs * h_ut * self.carrier_frequency / SPEED_OF_LIGHT
        )

        ## Basic path loss for LoS

        pl_1 = (
            20.0 * torch.log10(40.0 * PI * distance_3d * fc / 3.0)
            + torch.minimum(
                0.03 * torch.pow(average_building_height, 1.72),
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
            )
            * torch.log10(distance_3d)
            - torch.minimum(
                0.044 * torch.pow(average_building_height, 1.72),
                torch.tensor(14.77, dtype=self.dtype, device=self.device),
            )
            + 0.002 * torch.log10(average_building_height) * distance_3d
        )
        pl_2 = (
            20.0 * torch.log10(40.0 * PI * distance_breakpoint * fc / 3.0)
            + torch.minimum(
                0.03 * torch.pow(average_building_height, 1.72),
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
            )
            * torch.log10(distance_breakpoint)
            - torch.minimum(
                0.044 * torch.pow(average_building_height, 1.72),
                torch.tensor(14.77, dtype=self.dtype, device=self.device),
            )
            + 0.002 * torch.log10(average_building_height) * distance_breakpoint
            + 40.0 * torch.log10(distance_3d / distance_breakpoint)
        )
        pl_los = torch.where(distance_2d < distance_breakpoint, pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I

        pl_3 = (
            161.04
            - 7.1 * torch.log10(self.average_street_width)
            + 7.5 * torch.log10(average_building_height)
            - (24.37 - 3.7 * torch.square(average_building_height / h_bs))
            * torch.log10(h_bs)
            + (43.42 - 3.1 * torch.log10(h_bs)) * (torch.log10(distance_3d) - 3.0)
            + 20.0 * torch.log10(fc)
            - (3.2 * torch.square(torch.log10(11.75 * h_ut)) - 4.97)
        )
        pl_nlos = torch.maximum(pl_los, pl_3)

        ## Set the basic pathloss according to UT state

        # LoS
        pl_b = torch.where(self.los, pl_los, pl_nlos)

        self._update_attr("_pl_b", pl_b)
