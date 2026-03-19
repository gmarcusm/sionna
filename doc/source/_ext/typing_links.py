#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sphinx extension: rewrite typing.List/Dict links to stdtypes.

Makes cross-references that point to the typing module (e.g. typing.List,
typing.Dict) point to the built-in types docs instead (stdtypes.html#list,
stdtypes.html#dict).
"""

from docutils import nodes


# Map typing module fragment to (replacement path#fragment, description)
_TYPING_TO_STDTYPES = {
    "typing.html#typing.List": ("stdtypes.html#list", "list"),
    "typing.html#typing.Dict": ("stdtypes.html#dict", "dict"),
}


def _rewrite_typing_links(app, doctree, docname):
    """Rewrite reference nodes from typing.* to stdtypes.*."""
    for node in doctree.traverse(nodes.reference):
        refuri = node.get("refuri") or ""
        for typing_part, (stdtypes_part, _) in _TYPING_TO_STDTYPES.items():
            if typing_part in refuri:
                # Preserve base URL (e.g. https://docs.python.org/3/library/)
                new_refuri = refuri.replace(typing_part, stdtypes_part)
                node["refuri"] = new_refuri
                break


def setup(app):
    app.connect("doctree-resolved", _rewrite_typing_links)
    return {"version": "0.1", "parallel_read_safe": True}
