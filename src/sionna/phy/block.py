#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Definition of Sionna Block"""

from typing import Any, Optional
from .object import Object
from .config import Precision
import torch

__all__ = ["Block"]


class Block(Object):
    """Abstract class for Sionna PHY processing blocks.

    All Sionna PHY processing blocks inherit from this class. It provides
    automatic input casting to the block's precision, lazy building based
    on input shapes, and compatibility with ``torch.compile``.

    :param precision: Precision used for internal calculations and outputs.
        `None` (default) | ``"single"`` | ``"double"``.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
        Defaults to `None`.
    :param device: Device for computation (e.g., ``'cpu'``, ``'cuda:0'``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
        Defaults to `None`.
    """

    def __init__(
        self,
        *args: Any,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device)
        self._built = False

    @property
    def built(self) -> bool:
        """Indicates if the block's build function was called."""
        return self._built

    def build(self, *arg_shapes, **kwarg_shapes):
        r"""Initialize the block based on the inputs' shapes.

        Subclasses can override this method to create tensors or
        sub-blocks whose sizes depend on the input shapes.

        :param \*arg_shapes: Shapes of the positional arguments. Can be
            tuples (for tensors) or nested structures thereof (for
            lists/dicts of tensors).
        :param \*\*kwarg_shapes: Shapes of the keyword arguments. Can be
            tuples (for tensors) or nested structures thereof (for
            lists/dicts of tensors).
        """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the block, with setup code skipped during ``torch.compile`` tracing.

        When using ``torch.compile``, Dynamo traces this method and creates
        guards on state like ``_built``. To avoid recompilation when ``_built``
        changes from `False` to `True` after the first call, the setup code
        (input conversion and lazy build) is skipped when being traced. This
        is detected using ``torch.compiler.is_compiling()``.

        The setup code runs in eager mode (when ``is_compiling()`` is `False`),
        ensuring inputs are properly converted and blocks are built before the
        compiled ``forward()`` executes.
        """
        # Skip setup during Dynamo tracing to avoid guards on _built
        # Setup runs in eager mode on actual execution
        if not torch.compiler.is_compiling():
            args = self._convert(args)
            kwargs = self._convert(kwargs)

            if not self._built:
                arg_shapes = self._get_shape(args)
                kwarg_shapes = self._get_shape(kwargs)
                self.build(*arg_shapes, **kwarg_shapes)
                self._built = True

        return super().__call__(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegates to :meth:`call`."""
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Process inputs. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement 'call'.")
