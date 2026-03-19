#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Pytest configuration for NR unit tests."""

import pytest

from sionna.phy import config


@pytest.fixture(params=["single", "double"])
def precision(request):
    """Fixture that tests with both single and double precision."""
    original = config.precision
    config.precision = request.param
    yield request.param
    config.precision = original


@pytest.fixture
def single_precision():
    """Fixture for single precision."""
    original = config.precision
    config.precision = "single"
    yield "single"
    config.precision = original

