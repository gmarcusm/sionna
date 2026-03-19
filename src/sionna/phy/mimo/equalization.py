#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to MIMO channel equalization."""

from typing import Optional, Tuple
import torch

from sionna.phy.config import config, dtypes, Precision
from sionna.phy.utils import expand_to_rank, matrix_pinv
from sionna.phy.mimo.utils import whiten_channel

__all__ = [
    "lmmse_matrix",
    "lmmse_equalizer",
    "zf_equalizer",
    "mf_equalizer",
]


def lmmse_matrix(
    h: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""MIMO LMMSE Equalization matrix.

    This function computes the LMMSE equalization matrix for a MIMO link,
    assuming the following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    This function returns the LMMSE equalization matrix:

    .. math::

        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.


    If :math:`\mathbf{S}=\mathbf{I}_M`, a numerically more stable version of the equalization matrix is computed:

    .. math::

        \mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H} + \mathbf{I}\right)^{-1}\mathbf{H}^{\mathsf{H}} .

    :param h: Channel matrices with shape [..., M, K]
    :param s: Noise covariance matrices with shape [..., M, M].
        If `None`, the noise is assumed to be white with unit variance.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output g: [..., K, M], `torch.complex`. LMMSE equalization matrices.

    .. rubric:: Examples

    .. code-block:: python

        h = torch.complex(torch.randn(4, 2), torch.randn(4, 2))
        g = lmmse_matrix(h)
        # g.shape = torch.Size([2, 4])
    """
    # Determine dtype
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    h = h.to(dtype=cdtype)
    s_none = s is None

    if s is not None:
        s = s.to(dtype=cdtype)
    else:
        # Identity matrix with proper batch dimensions
        k = h.shape[-1]
        s = torch.eye(k, dtype=cdtype, device=h.device)
        s = expand_to_rank(s, h.dim(), 0)

    if not s_none:
        # Compute g = h^H @ (h @ h^H + s)^-1
        # hhs = h @ h^H + s
        # Note that hhs^H = hhs, hence it admits a Cholesky decomposition
        hhs = h @ h.mH + s

        # Solve hhs @ g_t = h for g_t using Cholesky
        # Use cholesky_ex with check_errors=False for CUDA graph compatibility
        chol, _ = torch.linalg.cholesky_ex(hhs, check_errors=False)
        # cholesky_solve: solve L L^H x = b
        # First solve L y = h, then L^H g_t = y
        y = torch.linalg.solve_triangular(chol, h, upper=False)
        g_t = torch.linalg.solve_triangular(chol.mH, y, upper=True)

        # Compute g = g_t^H = (hhs^-1 @ h)^H = h^H @ hhs^-1
        g = g_t.mH
    else:
        # Compute g = (h^H @ h + I)^-1 @ h^H
        hhs = h.mH @ h + s
        # Use cholesky_ex with check_errors=False for CUDA graph compatibility
        chol, _ = torch.linalg.cholesky_ex(hhs, check_errors=False)
        h_h = h.mH
        y = torch.linalg.solve_triangular(chol, h_h, upper=False)
        g = torch.linalg.solve_triangular(chol.mH, y, upper=True)

    return g


def lmmse_equalizer(
    y: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    whiten_interference: bool = True,
    precision: Optional[Precision] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""MIMO LMMSE Equalizer.

    This function implements LMMSE equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Lemma B.19) :cite:p:`BHS2017` :

    .. math::

        \hat{\mathbf{x}} = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}\mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of

    .. math::

        \mathop{\text{diag}}\left(\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]\right)
        = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H} \right)^{-1} - \mathbf{I}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}`
    is important for the :class:`~sionna.phy.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    :param y: Received signals with shape [..., M]
    :param h: Channel matrices with shape [..., M, K]
    :param s: Noise covariance matrices with shape [..., M, M]
    :param whiten_interference: If `True`, the interference is first whitened
        before equalization. In this case, an alternative expression for the
        receive filter is used that can be numerically more stable.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output x_hat: [..., K], `torch.complex`. Estimated symbol vectors.
    :output no_eff: [..., K], `torch.float`. Effective noise variance for each stream.

    .. rubric:: Examples

    .. code-block:: python

        y = torch.complex(torch.randn(4), torch.randn(4))
        h = torch.complex(torch.randn(4, 2), torch.randn(4, 2))
        s = torch.eye(4, dtype=torch.complex64)
        x_hat, no_eff = lmmse_equalizer(y, h, s)
    """
    # Determine dtype
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    y = y.to(dtype=cdtype)
    h = h.to(dtype=cdtype)
    s = s.to(dtype=cdtype)

    # LMMSE estimate of x:
    # x_hat = diag(GH)^(-1) @ G @ y
    # with G = H^H @ (H @ H^H + S)^(-1)
    #
    # This leads to the per-symbol model:
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # diag(E[ee^H]) = diag(GH)^(-1) - I

    if not whiten_interference:
        # Compute equalizer matrix G
        g = lmmse_matrix(h, s, precision=precision)
    else:
        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False)

        # Compute equalizer matrix G
        g = lmmse_matrix(h, s=None, precision=precision)

    # Compute G @ y
    gy = (g @ y.unsqueeze(-1)).squeeze(-1)

    # Compute G @ H
    gh = g @ h

    # Compute diag(G @ H)
    d = torch.diagonal(gh, dim1=-2, dim2=-1)

    # Compute x_hat = diag(G @ H)^-1 @ G @ y
    x_hat = gy / d

    # Compute residual error variance
    no_eff = (1.0 / d - 1.0).real

    return x_hat, no_eff


def zf_equalizer(
    y: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    precision: Optional[Precision] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""MIMO ZF Equalizer.

    This function implements zero-forcing (ZF) equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.10) :cite:p:`BHS2017` :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    :param y: Received signals with shape [..., M]
    :param h: Channel matrices with shape [..., M, K]
    :param s: Noise covariance matrices with shape [..., M, M]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output x_hat: [..., K], `torch.complex`. Estimated symbol vectors.
    :output no_eff: [..., K], `torch.float`. Effective noise variance for each stream.

    .. rubric:: Examples

    .. code-block:: python

        y = torch.complex(torch.randn(8), torch.randn(8))
        h = torch.complex(torch.randn(8, 4), torch.randn(8, 4))
        s = torch.eye(8, dtype=torch.complex64)
        x_hat, no_eff = zf_equalizer(y, h, s)
    """
    # Determine dtype
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    y = y.to(dtype=cdtype)
    h = h.to(dtype=cdtype)
    s = s.to(dtype=cdtype)

    # ZF estimate of x:
    # x_hat = G @ y
    # with G = (H^H @ H)^(-1) @ H^H (Moore-Penrose pseudoinverse)
    #
    # This leads to the per-symbol model:
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # E[e @ e^H] = G @ S @ G^H

    # Compute G (pseudoinverse)
    g = matrix_pinv(h)

    # Compute x_hat
    x_hat = (g @ y.unsqueeze(-1)).squeeze(-1)

    # Compute residual error variance
    gsg = g @ s @ g.mH
    no_eff = torch.diagonal(gsg, dim1=-2, dim2=-1).real

    return x_hat, no_eff


def mf_equalizer(
    y: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    precision: Optional[Precision] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""MIMO Matched Filter (MF) Equalizer.

    This function implements matched filter (MF) equalization for a
    MIMO link, assuming the following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Eq. 4.11) :cite:p:`BHS2017` :

    .. math::

        \hat{\mathbf{x}} = \mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}\mathbf{H}^{\mathsf{H}}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of the matrix

    .. math::

        \mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]
        = \left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)\left(\mathbf{I}-\mathbf{G}\mathbf{H} \right)^{\mathsf{H}} + \mathbf{G}\mathbf{S}\mathbf{G}^{\mathsf{H}}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{H}^{\mathsf{H}}\mathbf{H}\right)^{-1}`
    in the definition of :math:`\mathbf{G}`
    is important for the :class:`~sionna.phy.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    :param y: Received signals with shape [..., M]
    :param h: Channel matrices with shape [..., M, K]
    :param s: Noise covariance matrices with shape [..., M, M]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output x_hat: [..., K], `torch.complex`. Estimated symbol vectors.
    :output no_eff: [..., K], `torch.float`. Effective noise variance for each stream.

    .. rubric:: Examples

    .. code-block:: python

        y = torch.complex(torch.randn(8), torch.randn(8))
        h = torch.complex(torch.randn(8, 4), torch.randn(8, 4))
        s = torch.eye(8, dtype=torch.complex64)
        x_hat, no_eff = mf_equalizer(y, h, s)
    """
    # Determine dtype
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    y = y.to(dtype=cdtype)
    h = h.to(dtype=cdtype)
    s = s.to(dtype=cdtype)

    # MF estimate of x:
    # x_hat = G @ y
    # with G = diag(H^H @ H)^-1 @ H^H
    #
    # This leads to the per-symbol model:
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # E[e @ e^H] = (I - G @ H) @ (I - G @ H)^H + G @ S @ G^H

    # Compute G
    hth = h.mH @ h
    d = torch.diag_embed(1.0 / torch.diagonal(hth, dim1=-2, dim2=-1))
    g = d @ h.mH

    # Compute x_hat
    x_hat = (g @ y.unsqueeze(-1)).squeeze(-1)

    # Compute residual error variance
    gsg = g @ s @ g.mH
    gh = g @ h
    k = gh.shape[-1]
    i = torch.eye(k, dtype=gsg.dtype, device=gsg.device)
    i = expand_to_rank(i, gsg.dim(), 0)

    i_minus_gh = i - gh
    no_eff = torch.abs(torch.diagonal(i_minus_gh @ i_minus_gh.mH + gsg, dim1=-2, dim2=-1))

    return x_hat, no_eff

