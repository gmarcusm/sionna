#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Urban macrocell (UMa) channel model from 3GPP TR38.901 specification"""

from typing import Optional

from .system_level_channel import SystemLevelChannel
from .uma_scenario import UMaScenario
from .antenna import PanelArray

__all__ = ["UMa"]


class UMa(SystemLevelChannel):
    r"""
    Urban macrocell (UMa) channel model from 3GPP :cite:p:`TR38901` specification.

    Setting up a UMa model requires configuring the network topology, i.e., the
    UTs and BSs locations, UTs velocities, etc. This is achieved using the
    :meth:`~sionna.phy.channel.tr38901.UMa.set_topology` method. Setting a
    different topology for each batch example is possible. The batch size used
    when setting up the network topology is used for the link simulations.

    :param carrier_frequency: Carrier frequency [Hz]
    :param o2i_model: Outdoor-to-indoor loss model for UTs located indoor.
        Set this parameter to ``"low"`` to use the low-loss model, or to
        ``"high"`` to use the high-loss model.
        See section 7.4.3 of :cite:p:`TR38901` for details.
    :param ut_array: Panel array used by the UTs. All UTs share the same
        antenna array configuration.
    :param bs_array: Panel array used by the BSs. All BSs share the same
        antenna array configuration.
    :param direction: Link direction. Either ``"uplink"`` or ``"downlink"``.
    :param enable_pathloss: If `True`, apply pathloss. Otherwise don't.
        Defaults to `True`.
    :param enable_shadow_fading: If `True`, apply shadow fading. Otherwise
        don't. Defaults to `True`.
    :param always_generate_lsp: If `True`, new large scale parameters (LSPs)
        are generated for every new generation of channel impulse responses.
        Otherwise, always reuse the same LSPs, except if the topology is
        changed. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input num_time_samples: `int`.
        Number of time samples.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :output a: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx, num_tx, num_paths], `torch.float`.
        Path delays [s].

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel.tr38901 import PanelArray, UMa

        # UT and BS panel arrays
        bs_array = PanelArray(num_rows_per_panel=4, num_cols_per_panel=4,
                              polarization='dual', polarization_type='cross',
                              antenna_pattern='38.901', carrier_frequency=3.5e9)
        ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                              polarization='single', polarization_type='V',
                              antenna_pattern='omni', carrier_frequency=3.5e9)

        # Instantiating UMa channel model
        channel_model = UMa(carrier_frequency=3.5e9,
                            o2i_model='low',
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction='uplink')

        # Setting up network topology
        channel_model.set_topology(ut_loc, bs_loc, ut_orientations,
                                   bs_orientations, ut_velocities, in_state)

        # Generate channel
        h, delays = channel_model(num_time_samples=100, sampling_frequency=1e6)
    """

    def __init__(
        self,
        carrier_frequency: float,
        o2i_model: str,
        ut_array: PanelArray,
        bs_array: PanelArray,
        direction: str,
        enable_pathloss: bool = True,
        enable_shadow_fading: bool = True,
        always_generate_lsp: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        # UMa scenario
        scenario = UMaScenario(
            carrier_frequency,
            o2i_model,
            ut_array,
            bs_array,
            direction,
            enable_pathloss,
            enable_shadow_fading,
            precision=precision,
            device=device,
        )

        super().__init__(scenario, always_generate_lsp, precision=precision, device=device)




