"""Pytest configuration, command-line fixtures, and device parametrization."""

import os

# Ensure AMD Navi 23 (RX 6600M) maps cleanly to ROCm gfx1030 code objects
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import pytest
import torch
from re_reason.utils.seeding import set_seed



def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds custom command line options to pytest."""
    parser.addoption(
        "--seed",
        action="store",
        default=42,
        type=int,
        help="Random seed for deterministic test execution.",
    )
    parser.addoption(
        "--device",
        action="store",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run tests against.",
    )


@pytest.fixture(autouse=True)
def configured_seed(request: pytest.FixtureRequest) -> int:
    """Fixture that automatically applies deterministic seeding before every test."""
    seed = request.config.getoption("--seed")
    set_seed(seed)
    return seed


@pytest.fixture
def execution_device(request: pytest.FixtureRequest) -> torch.device:
    """Fixture returning the single execution device requested via --device."""
    dev_opt = request.config.getoption("--device")
    if dev_opt == "cpu":
        return torch.device("cpu")
    elif dev_opt == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA/ROCm requested via --device cuda but not available.")
        return torch.device("cuda")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=["cpu", "cuda"])
def all_available_devices(request: pytest.FixtureRequest) -> torch.device:
    """Parametrized fixture that runs tests across CPU and GPU/ROCm when available."""
    dev_str = request.param
    if dev_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device not available on this host.")
    return torch.device(dev_str)
