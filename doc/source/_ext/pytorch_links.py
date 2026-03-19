#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sphinx extension: :torch: role to link to PyTorch API documentation.

Use in RST as::

    :torch:`torch.linalg.cholesky_ex`
    :torch:`torch.linalg.solve_triangular`

This generates links to https://pytorch.org/docs/stable/generated/<name>.html.
"""

from docutils import nodes

PYTORCH_DOCS_BASE = "https://pytorch.org/docs/stable"
GENERATED_PREFIX = "generated/"


def _torch_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Role that links to PyTorch stable API docs (generated/*.html)."""
    options = options or {}
    content = content or []
    ref = text.strip()
    if not ref.startswith("torch."):
        ref = f"torch.{ref}"
    url = f"{PYTORCH_DOCS_BASE}/{GENERATED_PREFIX}{ref}.html"
    node = nodes.reference(rawtext, ref, refuri=url, **options)
    return [node], []


def setup(app):
    app.add_role("torch", _torch_role)
    return {"version": "0.1", "parallel_read_safe": True}
