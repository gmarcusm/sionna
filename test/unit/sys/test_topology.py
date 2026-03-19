#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.topology"""

import numpy as np
import pytest
import torch

from sionna.phy.utils import flatten_dims
from sionna.sys.topology import HexGrid
from sys_utils import wraparound_dist_np


class TestHexagonalGrid:
    """Tests for the HexGrid class."""

    def test_hexagonal_grid(self, device, precision):
        """Checks that the centers are aligned with pre-computed ones."""
        grid = HexGrid(
            cell_radius=4,
            num_rings=3,
            center_loc=(-2, 3),
            precision=precision,
            device=device,
        )
        grid.cell_radius = 1
        grid.num_rings = 2
        grid.center_loc = (0, 0)

        centers_precomputed = np.array(
            [
                (0.0, 0.0),
                (-1.5, 0.8660254037844386),
                (0.0, 1.7320508075688772),
                (1.5, 0.8660254037844386),
                (1.5, -0.8660254037844386),
                (0.0, -1.7320508075688772),
                (-1.5, -0.8660254037844386),
                (-3.0, 1.7320508075688772),
                (-1.5, 2.598076211353316),
                (0.0, 3.4641016151377544),
                (1.5, 2.598076211353316),
                (3.0, 1.7320508075688772),
                (3.0, 0.0),
                (3.0, -1.7320508075688772),
                (1.5, -2.598076211353316),
                (0.0, -3.4641016151377544),
                (-1.5, -2.598076211353316),
                (-3.0, -1.7320508075688772),
                (-3.0, 0.0),
            ]
        )

        centers = grid.cell_loc.cpu().numpy()[:, :2]
        is_found = np.zeros(len(centers))

        for c in centers:
            dist_c_centers = np.linalg.norm(np.array([c]) - centers_precomputed, axis=1)
            closest_center = np.argmin(dist_c_centers)
            assert dist_c_centers[closest_center] < 1e-5, "Center not found among pre-computed ones"
            is_found[closest_center] = 1

        assert np.sum(is_found) == len(is_found), "Not all centers were found"

    def test_drop_uts(self, device, precision):
        """Validate UT locations from call method."""
        isd = 50
        bs_height = 10
        num_rings = 1

        grid = HexGrid(
            isd=isd,
            num_rings=num_rings,
            cell_height=bs_height,
            precision=precision,
            device=device,
        )

        num_ut_per_sector = 100

        min_bs_ut_dist_vec = [20, 20, 20]
        max_bs_ut_dist_vec = [30, 35, 40]
        min_ut_height_vec = [1, 9, 12]
        max_ut_height_vec = [2, 11, 15]

        assert len(np.unique([
            len(min_bs_ut_dist_vec),
            len(min_ut_height_vec),
            len(max_ut_height_vec),
        ])) == 1

        for ii in range(len(min_bs_ut_dist_vec)):
            # [batch_size, num_cells, 3, num_ut_per_sector, 3]
            ut_loc, *_ = grid(
                1,
                num_ut_per_sector,
                min_bs_ut_dist_vec[ii],
                max_bs_ut_dist=max_bs_ut_dist_vec[ii],
                min_ut_height=min_ut_height_vec[ii],
                max_ut_height=max_ut_height_vec[ii],
            )
            # [num_cells, num_ut_per_cell, 3]
            ut_loc = flatten_dims(ut_loc, num_dims=2, axis=2)[0, ::].cpu().numpy()

            cell_loc = grid.cell_loc.cpu().numpy()

            for cell in range(grid.num_cells):
                for ut in range(ut_loc.shape[1]):
                    ut_cell_dist_3d = np.linalg.norm(
                        cell_loc[cell, :] - ut_loc[cell, ut, :]
                    )
                    ut_cell_dist_2d = np.linalg.norm(
                        cell_loc[cell, :2] - ut_loc[cell, ut, :2]
                    )

                    # 2D UT-cell center distance must be at most ISD / sqrt(3)
                    assert ut_cell_dist_2d <= grid.isd.item() / np.sqrt(3), (
                        "2D distance exceeds ISD / sqrt(3)"
                    )

                    # 3D UT-cell center distance must be >= min_bs_ut_dist
                    assert ut_cell_dist_3d >= min_bs_ut_dist_vec[ii], (
                        "3D distance is less than min_bs_ut_dist"
                    )

                    # 3D UT-cell center distance must be <= max_bs_ut_dist
                    assert ut_cell_dist_3d <= max_bs_ut_dist_vec[ii], (
                        "3D distance exceeds max_bs_ut_dist"
                    )

    def test_wraparound(self, device, precision):
        """Validate wraparound method against its non-PyTorch version."""

        def drop_uts(isd, num_rings, batch_size, num_ut_per_sector,
                     min_bs_ut_dist, min_ut_height, max_ut_height):
            grid = HexGrid(isd=isd, num_rings=num_rings, precision=precision, device=device)
            ut_loc, cell_mirror_coord, wrap_dist_pt = grid(
                batch_size,
                num_ut_per_sector,
                min_bs_ut_dist,
                min_ut_height=min_ut_height,
                max_ut_height=max_ut_height,
            )
            return ut_loc, cell_mirror_coord, wrap_dist_pt, grid

        batch_size = 1
        num_ut_per_sector = 5
        min_bs_ut_dist = 20
        isd = 50
        min_ut_height = 1
        max_ut_height = 2
        num_rings = 1

        # Run drop_uts
        ut_loc, cell_mirror_coord, wrap_dist_pt, grid = drop_uts(
            isd, num_rings,
            batch_size, num_ut_per_sector,
            min_bs_ut_dist, min_ut_height, max_ut_height
        )

        # Flatten to [..., 3] for UT locations
        ut_loc = flatten_dims(ut_loc, num_dims=4, axis=0).cpu().numpy()
        # [..., num_cells, 3]
        cell_mirror_coord = flatten_dims(cell_mirror_coord, num_dims=4, axis=0).cpu().numpy()
        # [..., num_cells]
        wrap_dist_pt = flatten_dims(wrap_dist_pt, num_dims=4, axis=0).cpu().numpy()

        # Compare wraparound distance against the Numpy version
        for ut in range(ut_loc.shape[0]):
            wrap_dist_np_vec = wraparound_dist_np(grid, ut_loc[ut, :])
            for cell in range(grid.num_cells):
                wrap_dist_np1 = np.linalg.norm(
                    cell_mirror_coord[ut, cell, :] - ut_loc[ut, :]
                )
                assert abs(wrap_dist_np_vec[cell] - wrap_dist_np1) < 1e-5, (
                    f"Mismatch between numpy wraparound methods at cell {cell}"
                )
                assert abs(wrap_dist_np_vec[cell] - wrap_dist_pt[ut, cell]) < 1e-5, (
                    f"Mismatch between numpy and PyTorch at cell {cell}"
                )

    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_compiled(self, device, precision, mode):
        """Test that HexGrid works with torch.compile."""
        if device == "cpu" and mode == "reduce-overhead":
            pytest.skip("reduce-overhead mode not well supported on CPU")

        isd = 50
        num_rings = 1
        batch_size = 2
        num_ut_per_sector = 3
        min_bs_ut_dist = 20

        grid = HexGrid(isd=isd, num_rings=num_rings, precision=precision, device=device)

        # Compile the call method
        if mode != "default":
            compiled_call = torch.compile(grid.call, mode=mode)
        else:
            compiled_call = grid.call

        # Run compiled version
        ut_loc, cell_mirror_coord, wrap_dist = compiled_call(
            batch_size, num_ut_per_sector, min_bs_ut_dist
        )

        # Basic shape checks
        assert ut_loc.shape == (batch_size, grid.num_cells, 3, num_ut_per_sector, 3)
        assert cell_mirror_coord.shape == (
            batch_size, grid.num_cells, 3, num_ut_per_sector, grid.num_cells, 3
        )
        assert wrap_dist.shape == (
            batch_size, grid.num_cells, 3, num_ut_per_sector, grid.num_cells
        )
