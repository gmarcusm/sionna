#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sphinx extension: :mitsuba: and :drjit: roles to link to Mitsuba 3 and Dr.Jit API docs.

Use in RST or docstrings as::

    :mitsuba:`Float`
    :mitsuba:`Vector3f`
    :drjit:`scalar.Array3f`
    :drjit:`zeros`

Mitsuba links go to the API reference; Dr.Jit types go to the type reference,
Dr.Jit function names go to the main reference.

:type: and other :py:class: / :py:func: refs using the common aliases ``mi.*``
and ``dr.*`` (e.g. ``:type: :py:class:`mi.Point2u``) are resolved via the
missing-reference handler so they link to the same external docs.
"""

from docutils import nodes

MITSUBA_DOCS_BASE = "https://mitsuba.readthedocs.io/en/stable"
MITSUBA_API_REF = f"{MITSUBA_DOCS_BASE}/src/api_reference.html"

DRJIT_DOCS_BASE = "https://drjit.readthedocs.io/en/stable"
DRJIT_TYPE_REF = f"{DRJIT_DOCS_BASE}/type_ref.html"
DRJIT_REFERENCE = f"{DRJIT_DOCS_BASE}/reference.html"


def _mitsuba_url(target):
    """Return URL for a Mitsuba type/object (e.g. mitsuba.Point2u or Point2u)."""
    if not target.startswith("mitsuba."):
        target = f"mitsuba.{target}"
    return f"{MITSUBA_API_REF}#{target}"


def _drjit_url(target):
    """Return URL for a Dr.Jit type or function (e.g. drjit.scalar.Array3f or zeros)."""
    if not target.startswith("drjit."):
        target = f"drjit.{target}"
    if "." in target and target.split(".")[-1][0].isupper():
        return f"{DRJIT_TYPE_REF}#{target.replace('.', '-').lower()}"
    return f"{DRJIT_REFERENCE}#{target.replace('.', '-').lower()}"


def _missing_reference(app, env, node, contnode):
    """Resolve :py:class:`mi.Point2u` and :py:func:`dr.zeros` etc. to external docs."""
    if node.get("refdomain") != "py":
        return None
    reftarget = node.get("reftarget", "")
    if reftarget.startswith("mi."):
        url = _mitsuba_url("mitsuba." + reftarget[3:])
        ref = nodes.reference("", "", refuri=url)
        ref += contnode
        return ref
    if reftarget.startswith("dr."):
        url = _drjit_url("drjit." + reftarget[3:])
        ref = nodes.reference("", "", refuri=url)
        ref += contnode
        return ref
    return None


def _mitsuba_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Role that links to Mitsuba 3 API reference."""
    options = options or {}
    content = content or []
    ref = text.strip()
    ref_display = ref if ref.startswith("mitsuba.") else ref
    url = _mitsuba_url(ref)
    node = nodes.reference(rawtext, ref_display, refuri=url, **options)
    return [node], []


def _drjit_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Role that links to Dr.Jit type reference or main API reference."""
    options = options or {}
    content = content or []
    ref = text.strip()
    url = _drjit_url(ref)
    node = nodes.reference(rawtext, ref, refuri=url, **options)
    return [node], []


def setup(app):
    app.add_role("mitsuba", _mitsuba_role)
    app.add_role("drjit", _drjit_role)
    app.connect("missing-reference", _missing_reference)
    return {"version": "0.1", "parallel_read_safe": True}
