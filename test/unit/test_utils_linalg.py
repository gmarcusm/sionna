#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch

from sionna.phy.utils.linalg import inv_cholesky, matrix_pinv
from sionna.phy.config import dtypes


def _run_compiled(fn, mode, *args, **kwargs):
    """Run a compiled function, skip test if compilation fails due to env issues."""
    compiled_fn = torch.compile(fn, mode=mode)
    try:
        return compiled_fn(*args, **kwargs)
    except Exception as e:
        err_msg = str(e).lower()
        if "ptxas" in err_msg or "triton" in err_msg or "inductor" in err_msg:
            pytest.skip(f"torch.compile not available in this environment: {e}")
        raise


def _batched_eye(
    batch_shape: tuple[int, ...], dim: int, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Create an identity matrix broadcast to a batch shape."""
    eye = torch.eye(dim, dtype=dtype, device=device)
    return eye.expand(*batch_shape, dim, dim)


def _get_tol(dtype: torch.dtype) -> float:
    """Return comparison tolerance based on dtype precision."""
    single_precision = (torch.float16, torch.float32, torch.complex32, torch.complex64)
    return 1e-4 if dtype in single_precision else 1e-8


def test_inv_cholesky_docstring_example() -> None:
    """Docstring example should return the identity matrix."""
    a = torch.eye(2)
    result = inv_cholesky(a)
    expected = torch.eye(2, dtype=result.dtype)
    assert torch.allclose(result, expected)


def test_matrix_pinv_docstring_example() -> None:
    """Docstring example should match torch.linalg.pinv."""
    torch.manual_seed(0)
    a = torch.randn(4, 2)
    result = matrix_pinv(a)
    expected = torch.linalg.pinv(a)
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_inv_cholesky_batched(device, precision) -> None:
    """Batched inputs produce inverse Cholesky factors."""
    dtype = dtypes[precision]["torch"]["dtype"]
    torch.manual_seed(0)
    batch_shape = (2, 3)
    dim = 4

    matrix = torch.randn(*batch_shape, dim, dim, device=device, dtype=dtype)
    hermitian = matrix @ matrix.mH
    hermitian = hermitian + 0.5 * torch.eye(dim, dtype=dtype, device=device)

    inv_factor = inv_cholesky(hermitian)
    chol_factor = torch.linalg.cholesky(hermitian)
    identity = inv_factor @ chol_factor

    expected = _batched_eye(batch_shape, dim, dtype, device)
    tol = _get_tol(dtype)
    assert torch.allclose(identity, expected, atol=tol, rtol=tol)


def test_matrix_pinv_matches_torch(device, precision) -> None:
    """matrix_pinv should align with torch.linalg.pinv across batches."""
    dtype = dtypes[precision]["torch"]["dtype"]
    torch.manual_seed(1)
    batch_shape = (3,)
    rows, cols = 5, 3

    tensor = torch.randn(*batch_shape, rows, cols, device=device, dtype=dtype)
    # Ensure full column rank by adding identity to top-left block
    tensor[..., :cols, :] += torch.eye(cols, dtype=dtype, device=device)

    result = matrix_pinv(tensor)
    expected = torch.linalg.pinv(tensor)

    tol = _get_tol(dtype)
    assert torch.allclose(result, expected, atol=tol, rtol=tol)
    assert result.shape == expected.shape


def test_inv_cholesky_complex(device, precision) -> None:
    """Verify inv_cholesky works with complex-valued Hermitian matrices."""
    cdtype = dtypes[precision]["torch"]["cdtype"]
    torch.manual_seed(2)
    dim = 3

    real = torch.randn(dim, dim, device=device, dtype=cdtype)
    hermitian = real @ real.mH + torch.eye(dim, dtype=cdtype, device=device)

    inv_factor = inv_cholesky(hermitian)
    chol_factor = torch.linalg.cholesky(hermitian)
    identity = inv_factor @ chol_factor

    expected = torch.eye(dim, dtype=cdtype, device=device)
    tol = _get_tol(cdtype)
    assert torch.allclose(identity, expected, atol=tol, rtol=tol)


def test_matrix_pinv_complex(device, precision) -> None:
    """Verify matrix_pinv works with complex-valued matrices."""
    cdtype = dtypes[precision]["torch"]["cdtype"]
    torch.manual_seed(3)
    rows, cols = 4, 2

    tensor = torch.randn(rows, cols, device=device, dtype=cdtype)
    tensor[:cols, :] += torch.eye(cols, dtype=cdtype, device=device)

    result = matrix_pinv(tensor)
    expected = torch.linalg.pinv(tensor)

    tol = _get_tol(cdtype)
    assert torch.allclose(result, expected, atol=tol, rtol=tol)


def test_inv_cholesky_compiled(device, precision, mode) -> None:
    """Verify inv_cholesky works under torch.compile with various modes."""
    dtype = dtypes[precision]["torch"]["dtype"]
    torch.manual_seed(10)
    dim = 4

    matrix = torch.randn(dim, dim, device=device, dtype=dtype)
    hermitian = matrix @ matrix.mH + torch.eye(dim, dtype=dtype, device=device)

    result = _run_compiled(inv_cholesky, mode, hermitian)
    expected = inv_cholesky(hermitian)

    tol = _get_tol(dtype)
    assert torch.allclose(result, expected, atol=tol, rtol=tol)


def test_matrix_pinv_compiled(device, precision, mode) -> None:
    """Verify matrix_pinv works under torch.compile with various modes."""
    dtype = dtypes[precision]["torch"]["dtype"]
    torch.manual_seed(11)
    rows, cols = 5, 3

    tensor = torch.randn(rows, cols, device=device, dtype=dtype)
    tensor[:cols, :] += torch.eye(cols, dtype=dtype, device=device)

    result = _run_compiled(matrix_pinv, mode, tensor)
    expected = matrix_pinv(tensor)

    tol = _get_tol(dtype)
    assert torch.allclose(result, expected, atol=tol, rtol=tol)


def test_inv_cholesky_compiled_complex(device, precision, mode) -> None:
    """Verify inv_cholesky works under torch.compile with complex tensors."""
    cdtype = dtypes[precision]["torch"]["cdtype"]
    torch.manual_seed(12)
    dim = 3

    matrix = torch.randn(dim, dim, device=device, dtype=cdtype)
    hermitian = matrix @ matrix.mH + torch.eye(dim, dtype=cdtype, device=device)

    result = _run_compiled(inv_cholesky, mode, hermitian)
    expected = inv_cholesky(hermitian)

    tol = _get_tol(cdtype)
    assert torch.allclose(result, expected, atol=tol, rtol=tol)


def test_matrix_pinv_compiled_complex(device, precision, mode) -> None:
    """Verify matrix_pinv works under torch.compile with complex tensors."""
    cdtype = dtypes[precision]["torch"]["cdtype"]
    torch.manual_seed(13)
    rows, cols = 4, 2

    tensor = torch.randn(rows, cols, device=device, dtype=cdtype)
    tensor[:cols, :] += torch.eye(cols, dtype=cdtype, device=device)

    result = _run_compiled(matrix_pinv, mode, tensor)
    expected = matrix_pinv(tensor)

    tol = _get_tol(cdtype)
    assert torch.allclose(result, expected, atol=tol, rtol=tol)


def test_inv_cholesky_complex_batched(device, precision) -> None:
    """Verify inv_cholesky works with batched complex-valued Hermitian matrices."""
    cdtype = dtypes[precision]["torch"]["cdtype"]
    torch.manual_seed(14)
    batch_shape = (2, 3)
    dim = 4

    matrix = torch.randn(*batch_shape, dim, dim, device=device, dtype=cdtype)
    hermitian = matrix @ matrix.mH + torch.eye(dim, dtype=cdtype, device=device)

    inv_factor = inv_cholesky(hermitian)
    chol_factor = torch.linalg.cholesky(hermitian)
    identity = inv_factor @ chol_factor

    expected = _batched_eye(batch_shape, dim, cdtype, device)
    tol = _get_tol(cdtype)
    assert torch.allclose(identity, expected, atol=tol, rtol=tol)


def test_matrix_pinv_complex_batched(device, precision) -> None:
    """Verify matrix_pinv works with batched complex-valued matrices."""
    cdtype = dtypes[precision]["torch"]["cdtype"]
    torch.manual_seed(15)
    batch_shape = (2,)
    rows, cols = 5, 3

    tensor = torch.randn(*batch_shape, rows, cols, device=device, dtype=cdtype)
    tensor[..., :cols, :] += torch.eye(cols, dtype=cdtype, device=device)

    result = matrix_pinv(tensor)
    expected = torch.linalg.pinv(tensor)

    tol = _get_tol(cdtype)
    assert torch.allclose(result, expected, atol=tol, rtol=tol)
    assert result.shape == expected.shape
