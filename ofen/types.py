from __future__ import annotations

import dataclasses
from typing import Any, Union

import torch
from numpy.typing import NDArray

NDArrayOrTensor = Union[NDArray[Any], torch.Tensor]
ModelFeatures = dict[str, Union[NDArrayOrTensor, list[torch.Tensor], list[NDArray[Any]]]]
ModelOutputs = Union[NDArrayOrTensor, list[NDArrayOrTensor], dict[str, NDArrayOrTensor]]

HFModelKwargs = dict[str, Any]
HFTokenizerKwargs = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Overflow:
    """Represents an overflow condition in text processing.

    Attributes
    ----------
        index (int): The index where the overflow occurred.
        length (int): The length of the overflow.

    """

    index: int
    length: int
