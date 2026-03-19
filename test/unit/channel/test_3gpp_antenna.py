#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for 3GPP TR 38.901 antenna classes"""

import numpy as np
import torch

from sionna.phy import PI, SPEED_OF_LIGHT, dtypes
from sionna.phy.channel.tr38901 import (
    AntennaElement,
    AntennaPanel,
    PanelArray,
    Antenna,
    AntennaArray,
)


class TestAntennaElement:
    """Tests for the AntennaElement class"""

    def test_omni_pattern_returns_ones(self, device, precision):
        """Test that omnidirectional pattern returns 1.0 for all angles"""
        ant = AntennaElement(pattern="omni", precision=precision, device=device)

        theta = torch.linspace(0.01, PI - 0.01, 10, device=device)
        phi = torch.linspace(-PI + 0.01, PI - 0.01, 10, device=device)

        pattern = ant._radiation_pattern_omni(theta, phi)

        assert torch.allclose(pattern, torch.ones_like(pattern))

    def test_38901_pattern_peak_at_boresight(self, device, precision):
        """Test that 38.901 pattern has maximum at boresight (theta=pi/2, phi=0)"""
        ant = AntennaElement(pattern="38.901", precision=precision, device=device)

        # Boresight direction
        theta_boresight = torch.tensor([PI / 2], device=device)
        phi_boresight = torch.tensor([0.0], device=device)
        pattern_boresight = ant._radiation_pattern_38901(theta_boresight, phi_boresight)

        # Off-boresight direction
        theta_off = torch.tensor([PI / 4], device=device)
        phi_off = torch.tensor([PI / 4], device=device)
        pattern_off = ant._radiation_pattern_38901(theta_off, phi_off)

        assert pattern_boresight > pattern_off

    def test_field_polarization_vertical(self, device, precision):
        """Test that vertical polarization (slant_angle=0) gives f_phi=0"""
        ant = AntennaElement(pattern="omni", slant_angle=0.0, precision=precision, device=device)

        theta = torch.tensor([PI / 2], device=device)
        phi = torch.tensor([0.0], device=device)
        f_theta, f_phi = ant.field(theta, phi)

        # For slant_angle=0, f_phi should be 0
        assert torch.allclose(f_phi, torch.zeros_like(f_phi), atol=1e-6)
        # f_theta should be non-zero (equal to 1 for omni pattern)
        assert torch.allclose(f_theta, torch.ones_like(f_theta), atol=1e-6)

    def test_field_polarization_horizontal(self, device, precision):
        """Test that horizontal polarization (slant_angle=pi/2) gives f_theta=0"""
        dtype = dtypes[precision]["torch"]["dtype"]
        ant = AntennaElement(pattern="omni", slant_angle=PI / 2, precision=precision, device=device)

        theta = torch.tensor([PI / 2], dtype=dtype, device=device)
        phi = torch.tensor([0.0], dtype=dtype, device=device)
        f_theta, f_phi = ant.field(theta, phi)

        # For slant_angle=pi/2, f_theta should be ~0
        assert torch.allclose(f_theta, torch.zeros_like(f_theta), atol=1e-6)
        # f_phi should be non-zero (equal to 1 for omni pattern)
        assert torch.allclose(f_phi, torch.ones_like(f_phi), atol=1e-6)

    def test_output_dtype(self, device, precision):
        """Test that output matches configured precision"""
        dtype = dtypes[precision]["torch"]["dtype"]
        ant = AntennaElement(pattern="omni", precision=precision, device=device)

        theta = torch.tensor([PI / 2], dtype=dtype, device=device)
        phi = torch.tensor([0.0], dtype=dtype, device=device)
        f_theta, f_phi = ant.field(theta, phi)

        assert f_theta.dtype == dtype
        assert f_phi.dtype == dtype
        assert f_theta.device == torch.device(device)

    def test_pattern_property(self, device, precision):
        """Test that pattern property returns the correct value"""
        ant_omni = AntennaElement(pattern="omni", precision=precision, device=device)
        ant_38901 = AntennaElement(pattern="38.901", precision=precision, device=device)

        assert ant_omni.pattern == "omni"
        assert ant_38901.pattern == "38.901"


class TestAntennaPanel:
    """Tests for the AntennaPanel class"""

    def test_antenna_positions_shape_single_pol(self, device, precision):
        """Test that antenna positions have correct shape for single polarization"""
        num_rows, num_cols = 4, 2
        panel = AntennaPanel(
            num_rows=num_rows,
            num_cols=num_cols,
            polarization="single",
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        expected_num_ant = num_rows * num_cols
        assert panel.ant_pos.shape == (expected_num_ant, 3)

    def test_antenna_positions_shape_dual_pol(self, device, precision):
        """Test that antenna positions have correct shape for dual polarization"""
        num_rows, num_cols = 4, 2
        panel = AntennaPanel(
            num_rows=num_rows,
            num_cols=num_cols,
            polarization="dual",
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        expected_num_ant = num_rows * num_cols * 2  # Double for dual polarization
        assert panel.ant_pos.shape == (expected_num_ant, 3)

    def test_antenna_positions_on_yz_plane(self, device, precision):
        """Test that all antennas lie on the y-z plane (x=0)"""
        panel = AntennaPanel(
            num_rows=3,
            num_cols=3,
            polarization="single",
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        # All x coordinates should be 0
        assert torch.allclose(panel.ant_pos[:, 0], torch.zeros_like(panel.ant_pos[:, 0]))

    def test_properties(self, device, precision):
        """Test that panel properties are correctly set"""
        dtype = dtypes[precision]["torch"]["dtype"]
        panel = AntennaPanel(
            num_rows=4,
            num_cols=3,
            polarization="dual",
            vertical_spacing=0.7,
            horizontal_spacing=0.6,
            precision=precision,
            device=device,
        )

        assert panel.num_rows == 4
        assert panel.num_cols == 3
        assert panel.polarization == "dual"
        assert torch.isclose(panel.vertical_spacing, torch.tensor(0.7, dtype=dtype, device=device))
        assert torch.isclose(panel.horizontal_spacing, torch.tensor(0.6, dtype=dtype, device=device))


class TestPanelArray:
    """Tests for the PanelArray class"""

    def test_num_antennas_single_panel_single_pol(self, device, precision):
        """Test total antenna count for single panel, single polarization"""
        array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.num_ant == 16
        assert array.num_panels == 1
        assert array.num_panels_ant == 16

    def test_num_antennas_single_panel_dual_pol(self, device, precision):
        """Test total antenna count for single panel, dual polarization"""
        array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.num_ant == 32  # 16 * 2 for dual polarization
        assert array.num_panels == 1
        assert array.num_panels_ant == 32

    def test_num_antennas_multiple_panels(self, device, precision):
        """Test total antenna count for multiple panels"""
        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
            num_rows=2,
            num_cols=2,
            precision=precision,
            device=device,
        )

        # 2x2 elements per panel, dual polarization = 8 antennas per panel
        # 2x2 panels = 4 panels
        assert array.num_ant == 32
        assert array.num_panels == 4
        assert array.num_panels_ant == 8

    def test_antenna_positions_scaled_by_wavelength(self, device, precision):
        """Test that antenna positions are scaled by wavelength"""
        dtype = dtypes[precision]["torch"]["dtype"]
        carrier_frequency = 3e9
        wavelength = SPEED_OF_LIGHT / carrier_frequency

        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            element_vertical_spacing=0.5,
            element_horizontal_spacing=0.5,
            precision=precision,
            device=device,
        )

        # Maximum spacing between antennas in wavelengths should be 0.5
        # In meters, this should be 0.5 * wavelength
        max_y_diff = array.ant_pos[:, 1].max() - array.ant_pos[:, 1].min()
        expected_max_y = 0.5 * wavelength  # 2 columns, 0.5 spacing

        assert torch.isclose(max_y_diff, torch.tensor(expected_max_y, dtype=dtype, device=device), rtol=1e-5)

    def test_polarization_indices_single(self, device, precision):
        """Test that polarization indices are correct for single polarization"""
        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="H",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.ant_ind_pol1.shape[0] == 4
        # For single polarization, pol2 should be empty
        assert array._ant_ind_pol2.numel() == 0

    def test_polarization_indices_dual(self, device, precision):
        """Test that polarization indices are correct for dual polarization"""
        array = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        # Each polarization should have half the antennas
        assert array.ant_ind_pol1.shape[0] == 4
        assert array.ant_ind_pol2.shape[0] == 4

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        array = PanelArray(
            num_rows_per_panel=4,
            num_cols_per_panel=4,
            polarization='dual',
            polarization_type='VH',
            antenna_pattern='38.901',
            carrier_frequency=3.5e9,
            num_cols=2,
            panel_horizontal_spacing=3.,
            device=device,
        )

        # Should create without errors and have correct number of antennas
        # 4x4 elements, dual pol = 32 per panel, 2 panels = 64 total
        assert array.num_ant == 64

    def test_polarization_types(self, device, precision):
        """Test that different polarization types are handled correctly"""
        # Single V
        array_v = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_v.polarization_type == "V"

        # Single H
        array_h = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="single",
            polarization_type="H",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_h.polarization_type == "H"

        # Dual VH
        array_vh = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_vh.polarization_type == "VH"

        # Dual cross
        array_cross = PanelArray(
            num_rows_per_panel=2,
            num_cols_per_panel=2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )
        assert array_cross.polarization_type == "cross"


class TestAntenna:
    """Tests for the Antenna class (single element)"""

    def test_single_element(self, device, precision):
        """Test that Antenna creates a single-element array"""
        ant = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert ant.num_ant == 1
        assert ant.num_rows_per_panel == 1
        assert ant.num_cols_per_panel == 1

    def test_dual_pol_single_element(self, device, precision):
        """Test dual polarization creates two antenna elements"""
        ant = Antenna(
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert ant.num_ant == 2
        assert ant.ant_ind_pol1.shape[0] == 1
        assert ant.ant_ind_pol2.shape[0] == 1

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        ant = Antenna(
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=3.5e9,
            device=device,
        )
        assert ant.num_ant == 1

    def test_inheritance(self, device, precision):
        """Test that Antenna inherits from PanelArray"""
        ant = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert isinstance(ant, PanelArray)


class TestAntennaArray:
    """Tests for the AntennaArray class"""

    def test_array_dimensions(self, device, precision):
        """Test that AntennaArray has correct dimensions"""
        array = AntennaArray(
            num_rows=4,
            num_cols=4,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert array.num_ant == 32  # 4x4 * 2 polarizations
        assert array.num_rows_per_panel == 4
        assert array.num_cols_per_panel == 4

    def test_custom_spacing(self, device, precision):
        """Test that custom element spacing is applied"""
        dtype = dtypes[precision]["torch"]["dtype"]
        array = AntennaArray(
            num_rows=2,
            num_cols=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            vertical_spacing=0.7,
            horizontal_spacing=0.3,
            precision=precision,
            device=device,
        )

        assert torch.isclose(array.element_vertical_spacing, torch.tensor(0.7, dtype=dtype, device=device))
        assert torch.isclose(array.element_horizontal_spacing, torch.tensor(0.3, dtype=dtype, device=device))

    def test_docstring_example(self, device):
        """Test that the example from the docstring works correctly"""
        array = AntennaArray(
            num_rows=4,
            num_cols=4,
            polarization='dual',
            polarization_type='cross',
            antenna_pattern='38.901',
            carrier_frequency=3.5e9,
            device=device,
        )
        assert array.num_ant == 32

    def test_inheritance(self, device, precision):
        """Test that AntennaArray inherits from PanelArray"""
        array = AntennaArray(
            num_rows=2,
            num_cols=2,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=3.5e9,
            precision=precision,
            device=device,
        )

        assert isinstance(array, PanelArray)

