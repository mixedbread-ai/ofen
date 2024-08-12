from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ofen.common.utils import first
from ofen.enums import EncodingFormat

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray

    from ofen.types import NDArrayOrTensor


def pad_to_max_length(
    inputs: dict[str, NDArrayOrTensor] | NDArrayOrTensor,
    max_length: int,
    pad_token_id: int,
) -> dict[str, NDArrayOrTensor] | NDArrayOrTensor:
    """Pad the inputs to the maximum length with the pad token id.

    Args:
    ----
        inputs (Union[Dict[str, NDArrayOrTensor], NDArrayOrTensor]): The inputs to pad.
        max_length (int): The maximum length to pad to.
        pad_token_id (int): The pad token id to use.

    Returns:
    -------
        Union[Dict[str, NDArrayOrTensor], NDArrayOrTensor]: Padded inputs.

    Raises:
    ------
        ValueError: If inputs are neither a dictionary nor a tensor/array.

    """
    if isinstance(inputs, dict):
        return _pad_dict_inputs(inputs, max_length, pad_token_id)
    elif isinstance(inputs, (torch.Tensor, np.ndarray)):
        return _pad_tensor_or_array(inputs, max_length, pad_token_id)
    else:
        msg = "Inputs must be either a dictionary or a tensor/array."
        raise ValueError(msg)


def _pad_dict_inputs(
    inputs: dict[str, NDArrayOrTensor], max_length: int, pad_token_id: int
) -> dict[str, NDArrayOrTensor]:
    """Helper function to pad dictionary inputs."""
    is_tensor = isinstance(first(inputs.values()), torch.Tensor)
    for key, value in inputs.items():
        if value.shape[-1] < max_length:
            if is_tensor:
                pre_allocated = torch.full(
                    (value.shape[0], max_length), pad_token_id, dtype=value.dtype, device=value.device
                )
                inputs[key] = torch.cat((value, pre_allocated[:, value.shape[-1] :]), dim=-1)
            else:
                pre_allocated = np.full((value.shape[0], max_length - value.shape[-1]), pad_token_id, dtype=value.dtype)
                inputs[key] = np.concatenate((value, pre_allocated), axis=-1)
    return inputs


def _pad_tensor_or_array(inputs: NDArrayOrTensor, max_length: int, pad_token_id: int) -> NDArrayOrTensor:
    """Helper function to pad tensor or array inputs."""
    if isinstance(inputs, torch.Tensor):
        pre_allocated = torch.full(
            (inputs.shape[0], max_length), pad_token_id, dtype=inputs.dtype, device=inputs.device
        )
        return torch.cat((inputs, pre_allocated[:, inputs.shape[-1] :]), dim=-1)
    else:
        pre_allocated = np.full((inputs.shape[0], max_length), pad_token_id, dtype=inputs.dtype)
        return np.concatenate((inputs, pre_allocated[:, inputs.shape[-1] :]), axis=-1)


def l2_normalize(tensor: NDArrayOrTensor) -> NDArrayOrTensor:
    """Normalize the outputs to have a unit norm.

    Args:
    ----
        tensor (NDArrayOrTensor): The outputs to normalize.

    Returns:
    -------
        NDArrayOrTensor: The normalized outputs.

    Raises:
    ------
        ValueError: If the input is neither a torch.Tensor nor a numpy.ndarray.

    """
    if isinstance(tensor, torch.Tensor):
        norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
        return tensor / norm
    elif isinstance(tensor, np.ndarray):
        norm = np.linalg.norm(tensor, axis=-1, keepdims=True)
        return tensor / norm
    else:
        msg = "Input must be either a torch.Tensor or a numpy.ndarray."
        raise ValueError(msg)


def sigmoid(x: NDArrayOrTensor) -> NDArrayOrTensor:
    """Apply the sigmoid function to the input.

    Args:
    ----
        x (NDArrayOrTensor): The input to apply sigmoid on.

    Returns:
    -------
        NDArrayOrTensor: The result after applying the sigmoid function.

    Raises:
    ------
        ValueError: If the input is neither a torch.Tensor nor a numpy.ndarray.

    """
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    else:
        msg = "Input must be either a torch.Tensor or a numpy.ndarray."
        raise ValueError(msg)


def quantize_embeddings(
    embeddings: NDArray,
    encoding_format: EncodingFormat | str,
    ranges: NDArray | None = None,
    calibration_embeddings: NDArray | None = None,
) -> NDArray:
    """Quantize embeddings to a lower precision to reduce memory footprint and increase speed.

    Args:
    ----
        embeddings (NDArray): Embeddings with dtype float32 to convert.
        encoding_format (Union[EncodingFormat, str]): The encoding format to convert the embeddings to.
        ranges (Optional[NDArray]): The start values for the quantization. Defaults to minimum values of the embeddings.
        calibration_embeddings (Optional[NDArray]): Embeddings used for calibration if ranges are not provided.

    Returns:
    -------
        NDArray: Quantized embeddings with the specified precision.

    Raises:
    ------
        ValueError: If the encoding format is not supported.

    """
    encoding_format = EncodingFormat(encoding_format)

    quantization_functions = {
        EncodingFormat.FLOAT: lambda: embeddings.astype(np.float32),
        EncodingFormat.FLOAT16: lambda: embeddings.astype(np.float16),
        EncodingFormat.INT8: lambda: _quantize_to_int(embeddings, encoding_format, ranges, calibration_embeddings),
        EncodingFormat.UINT8: lambda: _quantize_to_int(embeddings, encoding_format, ranges, calibration_embeddings),
        EncodingFormat.BINARY: lambda: (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(
            np.int8
        ),
        EncodingFormat.UBINARY: lambda: np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1),
        EncodingFormat.BASE64: lambda: np.array(
            [base64.b64encode(embedding.tobytes()).decode("utf-8") for embedding in embeddings]
        ),
    }

    if encoding_format in quantization_functions:
        return quantization_functions[encoding_format]()
    else:
        msg = f"Encoding format {encoding_format} not supported."
        raise ValueError(msg)


def _quantize_to_int(
    embeddings: NDArray,
    encoding_format: EncodingFormat,
    ranges: NDArray | None,
    calibration_embeddings: NDArray | None,
) -> NDArray:
    """Helper function to quantize embeddings to INT8 or UINT8."""
    if ranges is None:
        if calibration_embeddings is not None:
            ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
        else:
            ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))

    starts, ends = ranges[0, :], ranges[1, :]
    steps = (ends - starts) / 255

    with np.errstate(divide="ignore", invalid="ignore"):
        if encoding_format == EncodingFormat.INT8:
            return (((embeddings - starts) / steps) - 128).astype(np.int8)
        return ((embeddings - starts) / steps).astype(np.uint8)


def torch_or_np(item: Any) -> ModuleType:
    """Determine whether to use torch or numpy based on the input type.

    Args:
    ----
        item (Any): The input item to check.

    Returns:
    -------
        ModuleType: Either torch or numpy module.

    Raises:
    ------
        ValueError: If the input type is not supported.

    """
    if isinstance(item, (dict, list, tuple)):
        return torch_or_np(first(item.values()) if isinstance(item, dict) else item[0])
    elif isinstance(item, torch.Tensor):
        return torch
    elif isinstance(item, np.ndarray):
        return np
    else:
        msg = f"Unsupported input type: {type(item)}"
        raise ValueError(msg)


class TorchModule(torch.nn.Module):
    """A base class for PyTorch modules with additional utility methods."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        """Get the current device of the module."""
        return self._dummy.device

    def to(self, *args, **kwargs) -> TorchModule:
        """Move the module to the specified device or dtype."""
        return super().to(*args, **kwargs)

    def cuda(self, device_id: int | None = None) -> TorchModule:
        """Move the module to a CUDA device.

        Args:
        ----
            device_id (Optional[Union[int]]): The CUDA device to move to.

        Returns:
        -------
            TorchModule: The module moved to the specified CUDA device.

        Raises:
        ------
            RuntimeError: If CUDA is not available.

        """
        if not torch.cuda.is_available():
            msg = "CUDA is not available"
            raise RuntimeError(msg)
        return self.to(torch.device("cuda", device_id))

    def cpu(self) -> TorchModule:
        """Move the module to the CPU."""
        return self.to(torch.device("cpu"))

    def int8(self) -> TorchModule:
        """Convert the module to INT8."""
        if torch.backends.quantized.engine == "none":
            torch.backends.quantized.engine = "qnnpack"

        return torch.quantization.quantize_dynamic(self, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
