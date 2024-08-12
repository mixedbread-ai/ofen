from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from ofen.common.utils import ensure_import

if TYPE_CHECKING:
    import onnxruntime


class ModelType(int, Enum):
    """Enumeration of supported model types."""

    TEXT_ENCODER = 0
    IMAGE_ENCODER = 1
    CLIP_ENCODER = 2
    CROSS_ENCODER = 3


class PaddingStrategy(str, Enum):
    """Enumeration of padding strategies."""

    START = "start"
    END = "end"

    @classmethod
    def from_hf(cls, value: str) -> PaddingStrategy:
        """Convert Hugging Face padding strategy to PaddingStrategy.

        Args:
        ----
            value (str): Hugging Face padding strategy.

        Returns:
        -------
            PaddingStrategy: Corresponding PaddingStrategy enum.

        """
        return cls.START if value == "left" else cls.END


class PoolingStrategy(str, Enum):
    """Enumeration of pooling strategies."""

    MEAN_SQRT_LEN = "mean_sqrt_len"
    WEIGHTED_MEAN = "weighted_mean"
    MAX = "max"
    CLS = "cls"
    SUM = "sum"
    LAST = "last"
    MEAN = "mean"
    FIRST_LAST_AVG = "first_last_avg"
    NONE = "none"


class ActivationStrategy(str, Enum):
    """Enumeration of activation strategies for the activation layer."""

    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    TANH = "tanh"
    RELU = "relu"
    NONE = "none"


class NormalizationStrategy(str, Enum):
    """Enumeration of normalization strategies."""

    L2 = "l2"
    L1 = "l1"
    NONE = "none"


class EncodingFormat(str, Enum):
    """Enumeration of encoding formats."""

    FLOAT = "float"
    FLOAT16 = "float16"
    BASE64 = "base64"
    BINARY = "binary"
    UBINARY = "ubinary"
    INT8 = "int8"
    UINT8 = "uint8"


class TruncationStrategy(str, Enum):
    """Enumeration of truncation strategies."""

    NONE = "none"
    START = "start"
    END = "end"

    @classmethod
    def from_hf(cls, value: str) -> TruncationStrategy:
        """Convert Hugging Face truncation strategy to TruncationStrategy.

        Args:
        ----
            value (str): Hugging Face truncation strategy.

        Returns:
        -------
            TruncationStrategy: Corresponding TruncationStrategy enum.

        """
        return cls.START if value == "left" else cls.END


class ORTOptimizationLevel(str, Enum):
    """Enumeration of ONNX Runtime optimization levels."""

    O1 = "O1"
    O2 = "O2"
    O3 = "O3"
    O4 = "O4"

    def to_ort(self) -> onnxruntime.GraphOptimizationLevel | None:
        """Convert ORTOptimizationLevel to ONNX Runtime GraphOptimizationLevel.

        Returns
        -------
            Union[onnxruntime.GraphOptimizationLevel, None]: Corresponding ONNX Runtime optimization level.

        Raises
        ------
            ImportError: If onnxruntime is not installed.

        """
        with ensure_import("ofen[onnx]"):
            import onnxruntime as ort

        opt_levels = {
            ORTOptimizationLevel.O1: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            ORTOptimizationLevel.O2: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            ORTOptimizationLevel.O3: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            ORTOptimizationLevel.O4: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        return opt_levels.get(self)


class Quantization(str, Enum):
    """Enumeration of quantization types."""

    NONE = "none"
    INT8 = "int8"
    FP16 = "fp16"

    def __str__(self) -> str:
        """Return the string representation of the Quantization enum.

        Returns
        -------
            str: String representation of the enum value.

        """
        return self.value
