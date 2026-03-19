#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Test utilities for sionna.sys unit tests"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sionna.phy import Block, config, dtypes
from sionna.phy.config import Precision
from sionna.phy.mimo import StreamManagement
from sionna.phy.utils import db_to_lin, insert_dims, lin_to_db, sample_bernoulli
from sionna.sys import (
    InnerLoopLinkAdaptation,
    OuterLoopLinkAdaptation,
    PHYAbstraction,
    downlink_fair_power_control,
    get_pathloss,
    open_loop_uplink_power_control,
)
from sionna.sys.utils import spread_across_subcarriers


def compute_sinr_numpy(
    num_streams_per_rx: int,
    num_streams_per_tx: int,
    num_rx_per_tx: int,
    num_tx: int,
    thermal_noise_power: float,
    equalizer_sel: np.ndarray,
    precoding_sel: np.ndarray,
    rx_tx_association_sel: np.ndarray,
    stream_association_sel: np.ndarray,
    channel_sel: np.ndarray,
    tx_power_sel: np.ndarray,
    link_type: str,
) -> np.ndarray:
    """Numpy implementation of the per-stream SINR computation for a single RX.

    :param num_streams_per_rx: Number of streams per receiver
    :param num_streams_per_tx: Number of streams per transmitter
    :param num_rx_per_tx: Number of receivers per transmitter
    :param num_tx: Number of transmitters
    :param thermal_noise_power: Thermal noise power
    :param equalizer_sel: Equalizer for selected RX
    :param precoding_sel: Precoding matrices
    :param rx_tx_association_sel: RX-TX association for selected RX
    :param stream_association_sel: Stream association matrix
    :param channel_sel: Channel for selected RX
    :param tx_power_sel: TX power
    :param link_type: 'DL' for downlink, 'UL' for uplink
    :return: SINR per stream
    """
    signal_per_stream = np.zeros(num_streams_per_rx)
    interference_per_stream = np.zeros(num_streams_per_rx)
    noise_per_stream = np.zeros(num_streams_per_rx)

    for s in range(num_streams_per_rx):
        noise_per_stream[s] = thermal_noise_power * sum(
            np.power(abs(equalizer_sel[s, :]), 2)
        )

    if link_type == "DL":
        # TX attached to RX rx_sel
        tx_sel = np.where(rx_tx_association_sel == 1)[0][0]

        # stream for user to stream for TX
        s_rx_sel_to_tx = np.where(stream_association_sel[tx_sel, :] == 1)[0]

        # TX
        for tx in range(num_tx):
            H_b = channel_sel[tx, ::]
            # RX to which TX b transmits to
            for rx in range(num_rx_per_tx):
                # stream of user u, that TX b transmits to
                for s_rx_tx in range(num_streams_per_rx):
                    s_tx = rx * num_streams_per_rx + s_rx_tx
                    precoding_vector = precoding_sel[tx, :, s_tx][:, np.newaxis]
                    H_b_precoded = np.matmul(H_b, precoding_vector)
                    # stream in reception
                    for s_rx in range(num_streams_per_rx):
                        # received signal
                        y = np.matmul(equalizer_sel[s_rx, :], H_b_precoded)[0]
                        # signal power
                        y2 = abs(y) ** 2 * float(tx_power_sel[tx, s_tx])

                        is_intended_for_rx_sel = stream_association_sel[tx, s_tx] == 1
                        is_intended_for_s_rx = s_rx_sel_to_tx[s_rx] == s_tx

                        if is_intended_for_rx_sel & is_intended_for_s_rx:
                            signal_per_stream[s_rx] += y2
                        else:
                            interference_per_stream[s_rx] += y2

    elif link_type == "UL":
        # TX's attached to RX rx_sel
        tx_sel = np.where(rx_tx_association_sel == 1)[0]

        # TX
        for tx in range(num_tx):
            H_b = channel_sel[tx, ::]
            # stream in transmission
            for s_tx in range(num_streams_per_tx):
                precoding_vector = precoding_sel[tx, :, s_tx][:, np.newaxis]
                H_b_precoded = np.matmul(H_b, precoding_vector)
                # stream in reception
                for s_rx in range(num_streams_per_rx):
                    # received signal
                    y = np.matmul(equalizer_sel[s_rx, :], H_b_precoded)[0]
                    # signal power
                    y2 = abs(y) ** 2 * float(tx_power_sel[tx, s_tx])

                    is_intended_user = tx in tx_sel
                    if is_intended_user:
                        is_intended_stream_and_user = s_rx == (
                            list(tx_sel).index(tx) * num_streams_per_tx + s_tx
                        )
                    else:
                        is_intended_stream_and_user = False
                    if is_intended_stream_and_user:
                        signal_per_stream[s_rx] += y2
                    else:
                        interference_per_stream[s_rx] += y2

    # Compute SINR while setting 0/0 to 0
    ind0 = (signal_per_stream == 0) & (
        (interference_per_stream + noise_per_stream) == 0
    )
    sinr = np.zeros(signal_per_stream.shape)
    sinr[ind0] = 0
    sinr[~ind0] = signal_per_stream[~ind0] / (
        interference_per_stream[~ind0] + noise_per_stream[~ind0]
    )
    return sinr


def wraparound_dist_np(grid, point: np.ndarray) -> List[float]:
    """Non-PyTorch wraparound distance function between a point and a hexagon
    center within a spiral hexagonal grid.

    :param grid: HexGrid object
    :param point: Point coordinates [x, y, z]
    :return: List of wraparound distances to each cell
    """
    dist_to_bs = []
    for cell in grid.grid:
        # (x,y) coordinates of the centers of the 6 neighbors + current hexagon
        center_neighbors = np.array(
            [
                [grid.grid[cell].coord_euclid[i].item() + d[i].item() for i in [0, 1]]
                + [grid.cell_height.item()]
                for d in grid._mirror_displacements_euclid
            ]
        )

        # distance between point and centers of hexagons having the same
        # relative coordinates in neighboring cells
        dist_point_neighbors = [np.linalg.norm(point - c) for c in center_neighbors]
        dist_to_bs.append(min(dist_point_neighbors))
    return dist_to_bs


class MAC(Block):
    """OLLA + PHY abstraction + HARQ feedback Sionna Block."""

    def __init__(
        self,
        num_ut: int,
        bler_target: float,
        delta_up: float,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        self._phy_abs = PHYAbstraction(precision=precision, device=device)
        self._illa = InnerLoopLinkAdaptation(
            num_ut=num_ut,
            phy_abstraction=self._phy_abs,
            bler_target_init=bler_target,
            precision=precision,
            device=device,
        )
        self._olla = OuterLoopLinkAdaptation(
            num_ut=num_ut,
            inner_loop_link_adaptation=self._illa,
            delta_up=delta_up,
            precision=precision,
            device=device,
        )

    @property
    def olla(self) -> OuterLoopLinkAdaptation:
        """Get the OLLA object."""
        return self._olla

    def call(
        self,
        sinr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process SINR through link adaptation and PHY abstraction.

        :param sinr: Post-equalization SINR
        :return: Tuple of (mcs_index, tbler, harq_feedback, offset)
        """
        # Run outer loop link adaptation
        mcs_index, harq_feedback, sinr_eff, tbler, bler = self._olla(sinr)

        return mcs_index, tbler, harq_feedback, self._olla.sinr_offset


class SINREffFeedback(Block):
    """Generate SINR evolution and feedback."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        prob_feedback: float = 0.5,
        bounds: Tuple[float, float] = (5.0, 21.0),
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        self._bounds = torch.tensor(bounds, dtype=self.dtype, device=self.device)
        self._prob_feedback = prob_feedback
        self._shape = shape

        # Initialize true SINR value
        generator = None if torch.compiler.is_compiling() else self.torch_rng
        self.true_val_db = (
            torch.rand(shape, dtype=self.dtype, device=self.device, generator=generator)
            * (self._bounds[1] - self._bounds[0])
            + self._bounds[0]
        )

    def call(self) -> torch.Tensor:
        """Generate new SINR feedback.

        :return: SINR feedback (0 for unscheduled users)
        """
        generator = None if torch.compiler.is_compiling() else self.torch_rng

        # Update true value with random walk
        self.true_val_db = self.true_val_db + (
            torch.rand(
                self._shape, dtype=self.dtype, device=self.device, generator=generator
            )
            * 2
            - 1
        )
        self.true_val_db = torch.clamp(
            self.true_val_db, self._bounds[0], self._bounds[1]
        )

        # Generate feedback with probability
        p = sample_bernoulli(
            list(self._shape),
            self._prob_feedback,
            precision=self.precision,
            device=self.device,
        )
        sinr_feedback = torch.where(
            p,
            db_to_lin(self.true_val_db, precision=self.precision),
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
        )
        return sinr_feedback


def gen_num_allocated_re(
    prob_being_scheduled: float,
    shape: Tuple[int, ...],
    bounds: Tuple[int, int],
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Generate random number of allocated streams.

    :param prob_being_scheduled: Probability of being scheduled
    :param shape: Output shape
    :param bounds: Min/max bounds for allocated RE
    :param precision: Precision for computations
    :param device: Device for tensors
    :return: Number of allocated RE
    """
    if device is None:
        device = config.device

    generator = None if torch.compiler.is_compiling() else config.torch_rng(device)

    num_allocated_re = torch.randint(
        bounds[0],
        bounds[1],
        shape,
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    p = sample_bernoulli(list(shape), p=prob_being_scheduled, device=device)
    num_allocated_re = torch.where(
        p, num_allocated_re, torch.tensor(0, dtype=torch.int32, device=device)
    )
    return num_allocated_re


def get_stream_management(
    direction: str,
    num_rx: int,
    num_tx: int,
    num_streams_per_ut: int,
) -> Tuple[StreamManagement, int]:
    """Instantiate a StreamManagement object.

    It determines which data streams are intended for each receiver.

    :param direction: 'downlink' or 'uplink'
    :param num_rx: Number of receivers
    :param num_tx: Number of transmitters
    :param num_streams_per_ut: Number of streams per UT
    :return: Tuple of (stream_management, num_streams_per_tx)
    """
    if direction == "downlink":
        num_ut_per_sector = int(num_rx / num_tx)
        num_streams_per_tx = num_streams_per_ut * num_ut_per_sector

        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array(
            [
                [i1, i2]
                for i2 in range(num_tx)
                for i1 in np.arange(i2 * num_ut_per_sector, (i2 + 1) * num_ut_per_sector)
            ]
        )
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    else:
        num_ut_per_sector = int(num_tx / num_rx)
        num_streams_per_tx = num_streams_per_ut

        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array(
            [
                [i1, i2]
                for i1 in range(num_rx)
                for i2 in np.arange(i1 * num_ut_per_sector, (i1 + 1) * num_ut_per_sector)
            ]
        )
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    stream_management = StreamManagement(rx_tx_association, num_streams_per_tx)
    return stream_management, num_streams_per_tx


def get_pathloss_compiled(
    a: torch.Tensor,
    rx_tx_association: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compiled version of get_pathloss."""
    return get_pathloss(a, rx_tx_association=rx_tx_association, precision=precision)


def spread_across_subcarriers_compiled(
    tx_power_per_ut: torch.Tensor,
    is_scheduled: torch.Tensor,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    """Compiled version of spread_across_subcarriers."""
    return spread_across_subcarriers(
        tx_power_per_ut, is_scheduled, precision=precision
    )
