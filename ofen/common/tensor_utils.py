from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import numpy as np

from ofen.common.utils import first, try_import
from ofen.enums import EncodingFormat

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray

    from ofen.types import NDArrayOrTensor

with try_import("torch"):
    import torch

HAS_TORCH = torch is not None


def is_torch_tensor(x: Any) -> bool:
    """Check if the input is a torch.Tensor."""
    if not HAS_TORCH:
        msg = "Torch is not installed."
        raise ImportError(msg)
    return isinstance(x, torch.Tensor)


def is_numpy_array(x: Any) -> bool:
    """Check if the input is a numpy.ndarray."""
    return isinstance(x, np.ndarray)


def l2_normalize(tensor: NDArrayOrTensor) -> NDArrayOrTensor:
    """Normalize the outputs to have a unit norm.

    Args:
        tensor (NDArrayOrTensor): The outputs to normalize.

    Returns:
        NDArrayOrTensor: The normalized outputs.

    Raises:
        ValueError: If the input is neither a torch.Tensor nor a numpy.ndarray.

    """
    if is_numpy_array(tensor):
        norm = np.linalg.norm(tensor, axis=-1, keepdims=True)
        return tensor / norm
    if is_torch_tensor(tensor):
        return torch.nn.functional.normalize(tensor, p=2, dim=-1)

    msg = "Input must be either a torch.Tensor or a numpy.ndarray."
    raise ValueError(msg)


def sigmoid(x: NDArrayOrTensor) -> NDArrayOrTensor:
    """Apply the sigmoid function to the input.

    Args:
        x (NDArrayOrTensor): The input to apply sigmoid on.

    Returns:
        NDArrayOrTensor: The result after applying the sigmoid function.

    Raises:
        ValueError: If the input is neither a torch.Tensor nor a numpy.ndarray.

    """
    if is_numpy_array(x):
        return 1 / (1 + np.exp(-x))
    if is_torch_tensor(x):
        return torch.sigmoid(x)

    msg = "Input must be either a torch.Tensor or a numpy.ndarray."
    raise ValueError(msg)


def quantize_embeddings(
    embeddings: NDArrayOrTensor,
    encoding_format: EncodingFormat | str,
    factor: float | None = None,
) -> NDArrayOrTensor:
    """Quantize embeddings to a lower precision to reduce memory footprint and increase speed.

    Args:
        embeddings (NDArray): Embeddings with dtype float32 to convert.
        encoding_format (Union[EncodingFormat, str]): The encoding format to convert the embeddings to.

    Returns:
        NDArray: Quantized embeddings with the specified precision.

    Raises:
        ValueError: If the encoding format is not supported.

    """
    encoding_format = EncodingFormat(encoding_format)

    if encoding_format == EncodingFormat.FLOAT:
        return embeddings.astype(np.float32)
    if encoding_format == EncodingFormat.FLOAT16:
        return embeddings.astype(np.float16)
    if encoding_format == EncodingFormat.INT8 or encoding_format == EncodingFormat.UINT8:
        return _quantize_to_int(embeddings, encoding_format, factor)
    if encoding_format == EncodingFormat.BINARY:
        return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)
    if encoding_format == EncodingFormat.UBINARY:
        return np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)
    if encoding_format == EncodingFormat.BASE64:
        return np.array([base64.b64encode(embedding.tobytes()).decode("utf-8") for embedding in embeddings])

    msg = f"Encoding format {encoding_format} not supported."
    raise ValueError(msg)


def _quantize_to_int(
    embeddings: NDArray,
    encoding_format: EncodingFormat,
    factor: float | None = None,
) -> NDArray:
    """Helper function to quantize embeddings to INT8 or UINT8."""
    if factor is None:
        dim = embeddings.shape[-1]
        factor = 74.9 * np.log(dim) - 59.8

    clipped = np.clip(embeddings * factor, -127, 128)

    if encoding_format == EncodingFormat.INT8:
        return clipped.astype(np.int8)
    return (clipped + 128).astype(np.uint8)


def torch_or_np(item: Any) -> ModuleType:
    """Determine whether to use torch or numpy based on the input type.

    Args:
        item (Any): The input item to check.

    Returns:
        ModuleType: Either torch or numpy module.

    Raises:
        ValueError: If the input type is not supported.

    """
    if isinstance(item, (dict, list, tuple)):
        return torch_or_np(first(item.values()) if isinstance(item, dict) else item[0])
    if is_numpy_array(item):
        return np
    if is_torch_tensor(item):
        return torch
    msg = f"Unsupported input type: {type(item)}"
    raise ValueError(msg)
