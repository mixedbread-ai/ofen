from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ofen.enums import Quantization
    from ofen.types import Overflow


class QuantizationNotSupportedError(Exception):
    """Exception raised when quantization is not supported in a model.

    Attributes
    ----------
        model_type (str): The name of the model where the error occurred.
        quantization (Quantization): The quantization that is not supported.

    """

    def __init__(self, model_type: str, quantization: Quantization) -> None:
        """Initialize the QuantizationNotSupportedError with the given model type and quantization.

        Args:
        ----
            model_type (str): The type of the model where the error occurred.
            quantization (Quantization): The quantization that is not supported.

        """
        message = f"Quantization '{quantization}' is not supported for model type '{model_type}'."
        super().__init__(message)
        self.model_type = model_type
        self.quantization = quantization

    def __reduce__(self) -> tuple[Any, ...]:
        return self.__class__, (self.model_type, self.quantization)


class TruncationError(Exception):
    """Exception raised when truncation occurs in a model.

    Attributes
    ----------
        name_or_path (str): The name or path of the model where the error occurred.
        max_length (int): The maximum allowed length.
        overflows (List[Overflow]): The list of overflow instances.

    """

    def __init__(self, name_or_path: str, max_length: int, overflows: list[Overflow]) -> None:
        """Initialize the TruncationError with the given model name or path, maximum length, and overflows.

        Args:
        ----
            name_or_path (str): The name or path of the model where the error occurred.
            max_length (int): The maximum allowed length.
            overflows (List[Overflow]): The list of overflow instances.

        """
        message = (
            f"Truncation error for model '{name_or_path}' with max length '{max_length}'. " f"Overflowing: {overflows}"
        )
        super().__init__(message)
        self.name_or_path = name_or_path
        self.max_length = max_length
        self.overflows = overflows

    def __reduce__(self) -> tuple[Any, ...]:
        return self.__class__, (self.name_or_path, self.max_length, self.overflows)


class OrtExecutionProviderError(Exception):
    """Exception raised when a requested execution provider is not available."""

    def __init__(self, provider: str) -> None:
        """Initialize the OrtExecutionProviderError with the given provider.

        Args:
        ----
            provider (str): The name of the execution provider that is not available.

        """
        message = f"The requested execution provider '{provider}' is not available."
        super().__init__(message)
        self.provider = provider

    def __reduce__(self) -> tuple[Any, ...]:
        return self.__class__, (self.provider,)
