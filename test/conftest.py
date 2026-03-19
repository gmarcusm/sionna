#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Pytest configuration for Sionna test suite.

GPU Memory Management Tips for Running Full Test Suite:
--------------------------------------------------------
When running many GPU tests sequentially, CUDA memory can become fragmented,
leading to OOM errors even when individual tests pass. Options to mitigate:

1. Run with periodic garbage collection (default, configurable):
   pytest --gc-interval=25  # More aggressive cleanup every 25 tests

2. Run tests in smaller batches by test directory:
   pytest test/unit/channel/ && pytest test/unit/fec/ && ...

3. Use pytest-forked for maximum isolation (each test in separate process):
   pip install pytest-forked
   pytest --forked

4. Reduce parallelism if using pytest-xdist:
   pytest -n 1  # Single worker to reduce memory pressure

5. Set PYTORCH_ALLOC_CONF for better memory management:
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   (This is set automatically by conftest.py if not already set)
"""

import gc
import os
import pytest
import sys
import torch


def pytest_addoption(parser):
    """Add command line options for device selection."""
    parser.addoption(
        "--device",
        action="store",
        default="gpu",
        choices=["cpu", "gpu", "all"],
        help="Device to run tests on: cpu, gpu, or all (default: gpu)"
    )
    parser.addoption(
        "--gc-interval",
        action="store",
        default=50,
        type=int,
        help="Perform aggressive garbage collection every N tests (default: 50)"
    )


# Counter for periodic cleanup
_test_counter = 0


def pytest_configure(config) -> None:
    # Register custom markers
    config.addinivalue_line("markers", "gpu: mark test as GPU-only")

    # Configure PyTorch CUDA memory allocator for better memory management
    # during long test runs. This helps reduce memory fragmentation.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
        # expandable_segments helps reduce fragmentation by allowing
        # the allocator to release memory back to the system more easily
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Set SIONNA_DEVICE env var BEFORE importing sionna
    # This controls the default device for all tests
    device_option = config.getoption("--device", default="gpu")
    if device_option == "cpu":
        os.environ["SIONNA_DEVICE"] = "cpu"
    elif device_option == "gpu":
        os.environ["SIONNA_DEVICE"] = "cuda:0"
    # Note: sionna.phy.config should read SIONNA_DEVICE if set

    # Add test subdirectories to path for direct imports of test utilities
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for subdir in ["unit/channel", "unit/sys"]:
        path = os.path.join(test_dir, subdir)
        if path not in sys.path:
            sys.path.insert(0, path)

    # Add src directory to path for sionna imports
    src_dir = os.path.join(os.path.dirname(test_dir), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import sionna
    
    # Also set config.device directly after import
    if device_option == "cpu":
        sionna.phy.config.device = "cpu"
    elif device_option == "gpu" and torch.cuda.is_available():
        sionna.phy.config.device = "cuda:0"


@pytest.fixture(autouse=True)
def set_seed():
    import sionna.phy
    sionna.phy.config.seed = 42


def _clear_all_gpu_memory():
    """Helper to aggressively clear GPU memory on all available CUDA devices."""
    # Force Python garbage collection first (multiple passes for cyclic refs)
    gc.collect()
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                # Empty the CUDA cache
                torch.cuda.empty_cache()
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()
                # Reset memory stats to help with fragmentation tracking
                torch.cuda.reset_peak_memory_stats()


def _clear_compile_cache():
    """Clear torch.compile caches which can hold GPU memory."""
    try:
        torch._dynamo.reset()
    except Exception:
        pass  # Ignore if dynamo is not available or reset fails


@pytest.fixture(autouse=True)
def clear_gpu_memory(request):
    """
    Fixture that clears GPU memory before and after each test
    to prevent out-of-memory errors when running the full test suite.
    """
    global _test_counter
    _test_counter += 1

    gc_interval = request.config.getoption("--gc-interval", default=50)

    _clear_all_gpu_memory()
    yield
    # More aggressive cleanup after test
    _clear_all_gpu_memory()
    # Additional gc pass to catch any lingering references
    gc.collect()

    # Periodic extra-aggressive cleanup to combat fragmentation
    if _test_counter % gc_interval == 0 and torch.cuda.is_available():
        # Force a more thorough cleanup periodically
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset the CUDA memory allocator's internal state
        torch.cuda.reset_peak_memory_stats()


def pytest_runtest_teardown(item, nextitem):
    """
    Hook that runs after each test teardown.
    If the next test is in a different class or module (or there is no next test),
    perform aggressive memory cleanup to free class/module-scoped fixture data.
    """
    current_class = getattr(item, 'cls', None)
    next_class = getattr(nextitem, 'cls', None) if nextitem else None
    current_module = getattr(item, 'module', None)
    next_module = getattr(nextitem, 'module', None) if nextitem else None

    # If we're switching classes, modules, or finishing, do aggressive cleanup
    if current_class != next_class or current_module != next_module:
        # Clear torch.compile caches
        _clear_compile_cache()
        # Multiple GC passes
        gc.collect()
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def pytest_generate_tests(metafunc):
    """Generate test parameters based on --device option."""
    if "device" in metafunc.fixturenames:
        device_option = metafunc.config.getoption("--device")

        if device_option == "cpu":
            devices = ["cpu"]
        elif device_option == "gpu":
            if torch.cuda.is_available():
                devices = ["cuda:0"]
            else:
                pytest.skip("CUDA not available")
                devices = []
        else:  # "all"
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda:0")

        metafunc.parametrize("device", devices)


@pytest.fixture
def device(request):
    """
    Fixture that provides the device for testing.
    The actual parametrization is done by pytest_generate_tests.
    Also sets the global config.device for the duration of the test.
    """
    from sionna.phy import config
    device = request.param
    if device not in config.available_devices:
        pytest.skip(f"Device {device} not available")

    # Set global config device for this test
    original_device = config.device
    config.device = device
    yield device
    config.device = original_device

# List of precisions to test
PRECISIONS = ["single", "double"]
@pytest.fixture(params=PRECISIONS)
def precision(request):
    """
    Fixture that parametrizes tests over precisions.
    """
    return request.param

# List of compilation modes to test
MODES = ["default", "max-autotune", "reduce-overhead"]
@pytest.fixture(params=MODES)
def mode(request):
    """
    Fixture that parametrizes tests over compilation modes.
    """
    return request.param
