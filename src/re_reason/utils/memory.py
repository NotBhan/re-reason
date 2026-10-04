"""Memory diagnostic utilities for detecting leaks, graph retention, and reference accumulation."""

import gc
import sys
from typing import Any, Dict, Optional
import torch


def inspect_tensor_retention(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Inspects a tensor for autograd graph attachment and reference diagnostics.

    Args:
        tensor: The torch.Tensor to inspect, or None.

    Returns:
        Dictionary containing retention diagnostics.
    """
    if tensor is None:
        return {
            "is_none": True,
            "has_grad_fn": False,
            "requires_grad": False,
            "ref_count": 0,
        }

    return {
        "is_none": False,
        "has_grad_fn": tensor.grad_fn is not None,
        "requires_grad": tensor.requires_grad,
        "ref_count": sys.getrefcount(tensor),
        "is_leaf": tensor.is_leaf,
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "device": str(tensor.device),
    }


def get_memory_diagnostics(device: Optional[torch.device | str] = None) -> Dict[str, Any]:
    """Retrieves current memory diagnostics across GC and PyTorch caching allocators.

    Note: PyTorch allocator metrics (`torch.cuda.*`) are diagnostics only, as caching
    allocators retain reserved memory blocks by design.

    Args:
        device: Optional target device.

    Returns:
        Dictionary of memory diagnostic indicators.
    """
    dev_str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    diagnostics: Dict[str, Any] = {
        "gc_garbage_count": len(gc.garbage),
    }

    if "cuda" in dev_str or "hip" in dev_str:
        if torch.cuda.is_available():
            diagnostics["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            diagnostics["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
            diagnostics["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        else:
            diagnostics["cuda_allocated_bytes"] = 0
            diagnostics["cuda_reserved_bytes"] = 0
    else:
        diagnostics["cuda_allocated_bytes"] = 0
        diagnostics["cuda_reserved_bytes"] = 0

    return diagnostics


def check_monotonic_growth(measurements: list[int], tolerance_ratio: float = 0.05) -> bool:
    """Checks whether a sequence of memory measurements exhibits uncontrolled monotonic drift.

    Args:
        measurements: List of memory allocation measurements taken after warmup.
        tolerance_ratio: Allowable fluctuation ratio.

    Returns:
        True if measurements show uncontrolled monotonic growth, False if flat/bounded.
    """
    if len(measurements) < 10:
        return False

    # Check whether the latter half strictly drifts upwards beyond tolerance
    midpoint = len(measurements) // 2
    first_half_avg = sum(measurements[:midpoint]) / midpoint
    second_half_avg = sum(measurements[midpoint:]) / (len(measurements) - midpoint)

    if first_half_avg == 0:
        return second_half_avg > 0

    growth_ratio = (second_half_avg - first_half_avg) / first_half_avg
    return growth_ratio > tolerance_ratio
