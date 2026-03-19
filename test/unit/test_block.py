#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import torch
import numpy as np
from sionna.phy import Block, config, dtypes


class DummyBlock(Block):
    def call(self, *args, **kwargs):
        return args, kwargs


class DenseLayer(Block):
    """A simple Dense/Linear layer for testing purposes."""

    def __init__(self, units=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # We define a state variable to track if build was called
        self.build_counter = 0

    def build(self, x_shape):
        self.build_counter += 1
        input_dim = x_shape[-1]

        # Create weights as nn.Parameter for proper parameter tracking
        self.w = torch.nn.Parameter(torch.randn(
            (input_dim, self.units),
            device=self.device,
            dtype=self.dtype,
            generator=self.torch_rng,
        ))
        self.b = torch.nn.Parameter(torch.zeros(
            (self.units,),
            device=self.device,
            dtype=self.dtype,
        ))

    def call(self, x):
        return x @ self.w + self.b


class ComplexMultiplier(Block):
    """Test block for complex number handling."""

    def build(self, x_shape):
        self.factor = torch.tensor(
            1.0 + 1.0j, dtype=self.cdtype, requires_grad=True, device=self.device
        )

    def call(self, x):
        return x * self.factor


def test_block_without_call():
    """Test that calling Block without implementing call() raises NotImplementedError."""
    block = Block()
    with pytest.raises(NotImplementedError):
        block()


def test_block_no_inputs(device):
    block = DummyBlock(device=device)
    assert block() == ((), {})


def test_conversion(device):
    block = DummyBlock(device=device)
    args, kwargs = block(
        np.array([1.0, 2, 3]), 3, 4.0, 5.0j, a=1.0, b=2.0j, c=np.array([1j, 2j, 3j])
    )
    # numpy arrays are converted to tensors with block's dtype
    assert isinstance(args[0], torch.Tensor)
    assert args[0].dtype == block.dtype
    # ints stay as ints
    assert isinstance(args[1], int)
    # floats are converted to tensors with block's dtype
    assert isinstance(args[2], torch.Tensor)
    assert args[2].dtype == block.dtype
    # complex values are converted to tensors with block's cdtype
    assert isinstance(args[3], torch.Tensor)
    assert args[3].dtype == block.cdtype
    # Same for kwargs
    assert isinstance(kwargs["a"], torch.Tensor)
    assert kwargs["a"].dtype == block.dtype
    assert isinstance(kwargs["b"], torch.Tensor)
    assert kwargs["b"].dtype == block.cdtype
    assert kwargs["c"].dtype == block.cdtype


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((np.array([1.0, 2.0, 3.0]),), {}),
        ((torch.tensor([1.0, 2.0, 3.0]),), {}),
        (([1.0, 2.0, 3.0],), {}),
        (((1.0, 2.0, 3.0),), {}),
        ((1, 2.0, 3.0j), {}),
        ((np.array([1.0, 2.0]),), {"x": 3.0, "y": 4.0j}),
        ((), {"a": np.array([1.0]), "b": 123, "c": 1.0, "d": 2.0j}),
        (
            (np.array([1, 2, 3]), torch.tensor([4, 5, 6])),
            {"foo": np.array([7.7, 8.8]), "bar": 9},
        ),
        ((None,), {}),
        ((None,), {"a": "test"}),
        (([1, 2.2, 3.3j],), {}),
    ],
)
def test_compilation(device, mode, args, kwargs):
    block = DummyBlock(device=device)

    @torch.compile(mode=mode)
    def fun(*args, **kwargs):
        return block(*args, **kwargs)

    x, y = fun(*args, **kwargs)
    assert len(x) == len(args)
    assert len(y) == len(kwargs)


def test_lazy_initialization(device):
    """
    Verify that build() is called exactly once and variables are created.
    """
    model = DenseLayer(units=8, device=device)

    # Before call, model should not be built
    assert not model.built
    assert not hasattr(model, "w")

    # Create input
    x = torch.randn(2, 4, device="cpu")

    # First call
    y = model(x)

    # Checks
    assert model.built
    assert model.build_counter == 1
    assert hasattr(model, "w")
    assert y.shape == (2, 8)

    # Second call - ensure build is NOT called again
    model(x)
    assert model.build_counter == 1


def test_auto_type_casting(precision):
    """
    Verify that NumPy inputs and different types are cast to the block's dtype.
    """
    model = DenseLayer(units=4, precision=precision)
    x_np = np.random.randn(2, 4).astype(np.float32)
    y = model(x_np)

    assert isinstance(y, torch.Tensor)
    assert y.dtype == model.dtype
    assert model.w.dtype == model.dtype


def test_complex_support(precision, device):
    """
    Verify handling of complex numbers and cdtype.
    """
    model = ComplexMultiplier(precision=precision, device=device)

    # Real-valued input
    x = torch.randn(2, 2, dtype=torch.float32)  # Real input
    y = model(x)
    assert y.dtype == dtypes[precision]["torch"]["cdtype"]
    assert model.factor.dtype == dtypes[precision]["torch"]["cdtype"]

    # Let's pass a complex numpy array
    x_np_complex = np.array([[1 + 1j, 1 - 1j]], dtype=np.complex128)
    y = model(x_np_complex)
    assert y.dtype == dtypes[precision]["torch"]["cdtype"]
    assert model.factor.dtype == dtypes[precision]["torch"]["cdtype"]


def test_nested_inputs(device):
    """
    Verify the recursive structure handling (dict/list support).
    """

    class StructureBlock(Block):
        def build(self, inputs_shape):
            # Expecting a dictionary
            self.input_dim = inputs_shape["feature_a"][-1]
            self.w = torch.randn(
                self.input_dim,
                1,
                requires_grad=True,
                device=self.device,
                dtype=self.dtype,
            )

        def call(self, inputs):
            # Inputs should be a dictionary of tensors now
            return inputs["feature_a"] @ self.w + inputs["feature_b"]

    model = StructureBlock(device=device)

    # Input is a dictionary of numpy arrays
    data = {"feature_a": np.random.randn(10, 5), "feature_b": np.random.randn(10, 1)}

    out = model(data)

    assert model.built
    assert out.shape == (10, 1)
    assert isinstance(out, torch.Tensor)


def test_optimization_loop(device, precision):
    """
    Test that block recognizes parameters and optimizers can update them.
    """
    model = DenseLayer(units=1, device=device, precision=precision)
    x = torch.ones(1, 4)
    target = torch.zeros(1, 1, device=device, dtype=model.dtype)
    model(x)  # Trigger build
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Initial weights (save copy to compare)
    initial_w = model.w.clone()

    # Forward Pass
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, target)

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that weights changed
    assert not torch.allclose(model.w, initial_w)
    assert model.w.grad is not None


def test_build_compilation(device, mode):
    """
    Test that block builds are compiled.
    """
    model = DenseLayer(units=1, device=device)

    @torch.compile(mode=mode)
    def fun(x):
        return model(x)

    x = torch.ones(1, 4)
    y = fun(x)
    assert model.built

    y.norm().backward()
    assert torch.norm(model.w.grad) > 0
    assert torch.norm(model.b.grad) > 0
