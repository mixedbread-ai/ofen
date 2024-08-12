from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class MemoryMetrics:
    """Dataclass to store memory usage metrics for GPU operations.

    Attributes
    ----------
        memory_used (int): Amount of memory used in bytes.
        peak_memory (int): Peak memory usage in bytes.
        memory_per_input (float): Memory used per input in bytes.
        max_inputs (int): Maximum number of inputs that can be processed.
        optimal_batch_size (int): Optimal batch size for processing.
        max_tokens (int): Maximum number of tokens that can be processed.

    """

    memory_used: int
    peak_memory: int
    memory_per_input: float
    max_inputs: int
    optimal_batch_size: int
    max_tokens: int

    def __str__(self) -> str:
        """Return a formatted string representation of the memory metrics."""
        return (
            f"Memory Metrics:\n"
            f"  Memory Used: {self.memory_used / 1e6:.2f} MB\n"
            f"  Peak Memory: {self.peak_memory / 1e6:.2f} MB\n"
            f"  Memory per Input: {self.memory_per_input / 1e6:.2f} MB\n"
            f"  Max Inputs: {self.max_inputs:.0f}\n"
            f"  Optimal Batch Size: {self.optimal_batch_size}\n"
            f"  Max Tokens: {self.max_tokens}"
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Custom pretty representation for interactive environments."""
        if cycle:
            p.text("MemoryMetrics(...)")
        else:
            p.text(str(self))


def measure_memory_usage(
    model: nn.Module, pre_process: callable, example_input: list[str] | None = None, batch_size: int = 1
) -> MemoryMetrics:
    """Measure GPU memory usage for a given model and input.

    Args:
    ----
        model (nn.Module): The PyTorch model to measure.
        pre_process (callable): Function to preprocess the input.
        example_input (List[str]): Example input to use for measurement.
        batch_size (int): Batch size to use for measurement.

    Returns:
    -------
        MemoryMetrics: Object containing memory usage metrics.

    Raises:
    ------
        RuntimeError: If CUDA is not available.

    """
    if example_input is None:
        example_input = [". " * 8096]
    if not torch.cuda.is_available():
        msg = "CUDA is not available. This function requires a CUDA-enabled GPU."
        raise RuntimeError(msg)

    device = model.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Warm up the model
    with torch.no_grad():
        warm_up_input = pre_process(example_input)
        warm_up_input = {k: v.to(device) for k, v in warm_up_input.items()}
        _ = model(**warm_up_input)

    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    # Record initial memory usage
    torch.cuda.synchronize(device)
    initial_memory = torch.cuda.memory_allocated(device)

    # Create input tensor
    input_tensor = pre_process(example_input * batch_size)
    input_tensor = {k: v.to(device) for k, v in input_tensor.items()}

    # Perform forward pass
    with torch.no_grad():
        _ = model(**input_tensor)

    # Synchronize CUDA and record peak memory usage
    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device)

    # Calculate memory used
    memory_used = peak_memory - initial_memory

    torch.cuda.empty_cache()
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    static_memory = peak_memory - memory_used
    available_memory = total_gpu_memory - allocated_memory - reserved_memory - static_memory
    max_inputs = available_memory / (memory_used / batch_size)

    return MemoryMetrics(
        memory_used=memory_used,
        peak_memory=peak_memory,
        memory_per_input=memory_used / batch_size,
        max_inputs=max_inputs,
        optimal_batch_size=2 ** math.floor(math.log2(max_inputs)),
        max_tokens=math.floor(len(next(iter(input_tensor.values()))[0]) * max_inputs),
    )
