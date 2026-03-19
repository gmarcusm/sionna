#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sphinx extension: :input:/:output: domain fields and member section headings.

Registers :input name: and :output name: as proper Sphinx domain fields
on all Python directives (classes, functions, methods). They behave like
:param name: — grouped under "Input"/"Output" headings with type support
via :inputtype name: / :outputtype name: (or inline types).

Also inserts "Attributes" and "Methods" rubric headings in class
documentation when autodoc_member_order is "groupwise", and strips
the autodoc-generated "Return type" field (now redundant with :output:).
"""

from docutils import nodes
from sphinx import addnodes

from sphinx.domains.python import PyObject
from sphinx.util.docfields import TypedField


def _postprocess_doctree(app, doctree, docname):
    """Post-process the resolved doctree.

    - Insert 'Attributes' and 'Methods' rubric headings in class docs.
    - Strip autodoc-generated 'Return type' fields (redundant with :output:).
    """
    for class_desc in doctree.traverse(addnodes.desc):
        if class_desc.get("domain") != "py":
            continue
        if class_desc.get("objtype") not in ("class", "exception"):
            continue

        # Find the desc_content (class body)
        for content in class_desc.traverse(addnodes.desc_content):
            _inject_rubrics(content)
            break

    # Strip "Return type" fields from all field lists
    _strip_return_type_fields(doctree)


def _inject_rubrics(content):
    """Inject rubric nodes before attribute and method groups."""
    attr_types = {"attribute", "property"}
    method_types = {"method", "classmethod", "staticmethod"}

    first_attr = None
    first_method = None

    for child in content.children:
        if not isinstance(child, addnodes.desc):
            continue
        if child.get("domain") != "py":
            continue
        objtype = child.get("objtype", "")
        if objtype in attr_types and first_attr is None:
            first_attr = child
        elif objtype in method_types and first_method is None:
            first_method = child

    # Insert in reverse order so indices stay valid
    if first_method is not None:
        rubric = nodes.rubric("", "Methods")
        idx = list(content.children).index(first_method)
        content.insert(idx, rubric)

    if first_attr is not None:
        rubric = nodes.rubric("", "Attributes")
        idx = list(content.children).index(first_attr)
        content.insert(idx, rubric)


def _strip_return_type_fields(doctree):
    """Remove autodoc-generated 'Return type' fields from field lists.

    These are redundant now that outputs are documented via :output: fields.
    The 'Return type' is rendered by autodoc as a field list item with the
    field name 'Return type'.
    """
    for field_list in list(doctree.traverse(nodes.field_list)):
        to_remove = []
        for field in list(field_list.traverse(nodes.field)):
            field_name = field[0]  # first child is field_name node
            if field_name.astext().strip() == "Return type":
                to_remove.append(field)
        for field in to_remove:
            field.parent.remove(field)
        # If the field list is now empty, remove it too
        if len(field_list.children) == 0:
            field_list.parent.remove(field_list)


def setup(app):
    # Register :input: and :output: as typed domain fields
    PyObject.doc_field_types.append(
        TypedField(
            "input",
            label="Inputs",
            names=("input",),
            typerolename="class",
            typenames=("inputtype",),
            can_collapse=True,
        )
    )
    PyObject.doc_field_types.append(
        TypedField(
            "output",
            label="Outputs",
            names=("output",),
            typerolename="class",
            typenames=("outputtype",),
            can_collapse=True,
        )
    )

    # Post-process doctree: rubric headings + strip Return type fields
    app.connect("doctree-resolved", _postprocess_doctree)

    return {"version": "0.4", "parallel_read_safe": True}
