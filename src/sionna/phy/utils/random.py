#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Smart random number generation utilities for torch.compile compatibility.

These functions automatically switch between using a generator (for reproducibility
in eager mode) and global RNG state (for graph fusion in compiled mode).

Note: For proper multi-device reproducibility, use `sionna.phy.config.seed` instead
of `torch.manual_seed()`. The config seeds each device's default generator with a
device-specific offset, ensuring different devices produce different random streams.
"""

import math
from typing import Optional, Sequence, Union
import torch

from sionna.phy.config import config, dtypes, Precision

__all__ = ["randint", "rand", "uniform", "normal", "complex_normal"]


def randint(
    low: int,
    high: int,
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random integer tensor, compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param low: Minimum value (inclusive).
    :param high: Maximum value (exclusive).
    :param size: Shape of the output tensor.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor.
    :param generator: Random number generator (used only in eager mode).

    :output samples: Tensor with random integer values.
    """
    if torch.compiler.is_compiling():
        return torch.randint(low, high, size, dtype=dtype, device=device)
    else:
        return torch.randint(
            low, high, size, dtype=dtype, device=device, generator=generator
        )


def rand(
    size: Sequence[int],
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random uniform tensor [0, 1), compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param size: Shape of the output tensor.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor.
    :param generator: Random number generator (used only in eager mode).

    :output samples: Tensor with random uniform values.
    """
    if torch.compiler.is_compiling():
        return torch.rand(size, dtype=dtype, device=device)
    else:
        return torch.rand(size, dtype=dtype, device=device, generator=generator)


def uniform(
    size: Sequence[int],
    *,
    low: float = 0.0,
    high: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random uniform tensor in [low, high), compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param size: Shape of the output tensor.
    :param low: Lower bound (inclusive). Defaults to 0.0.
    :param high: Upper bound (exclusive). Defaults to 1.0.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor.
    :param generator: Random number generator (used only in eager mode).

    :output samples: Tensor with random uniform values in [low, high).
    """
    result = rand(size, dtype=dtype, device=device, generator=generator)
    if low != 0.0 or high != 1.0:
        result = result * (high - low) + low
    return result


def normal(
    size: Sequence[int],
    *,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random normal tensor, compile-aware.

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses torch.randn (which uses the Graph RNG) to ensure
    proper synchronization with other random operations like randint.

    Note: torch.normal uses a different RNG stream than torch.randn under
    torch.compile, which can cause training issues. Using torch.randn with
    scaling ensures consistent RNG behavior.

    :param size: Shape of the output tensor.
    :param mean: Mean of the distribution. Defaults to 0.0.
    :param std: Standard deviation of the distribution. Defaults to 1.0.
    :param dtype: Data type of the output tensor.
    :param device: Device for the output tensor.
    :param generator: Random number generator (used only in eager mode).

    :output samples: Tensor with random normal values.
    """
    if torch.compiler.is_compiling():
        # Use torch.randn which uses the Graph RNG (same as randint),
        # then apply mean and std transformation.
        # This avoids the RNG desynchronization issue with torch.normal.
        result = torch.randn(size, dtype=dtype, device=device)
        if std != 1.0:
            result = result * std
        if mean != 0.0:
            result = result + mean
        return result
    else:
        return torch.normal(
            mean=mean,
            std=std,
            size=size,
            dtype=dtype,
            device=device,
            generator=generator,
        )


def complex_normal(
    size: Sequence[int],
    *,
    precision: Optional[Precision] = None,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate complex normal random tensor with unit variance, compile-aware.

    Generates complex Gaussian random variables with variance 1 (i.e., variance 0.5
    per real and imaginary component).

    In eager mode, uses the provided generator for reproducibility.
    In compiled mode, uses global RNG state for graph fusion.

    :param size: Shape of the output tensor.
    :param precision: Precision used for the output tensor.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for the output tensor.
    :param generator: Random number generator (used only in eager mode).

    :output samples: Complex tensor with standard complex normal values.
    """
    # Determine dtype from precision
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    # Generate real and imaginary parts
    real = normal(size, dtype=dtype, device=device, generator=generator)
    imag = normal(size, dtype=dtype, device=device, generator=generator)
    # Scale by 1/sqrt(2) to get unit variance for the complex number
    return torch.complex(real, imag) * math.sqrt(0.5)
