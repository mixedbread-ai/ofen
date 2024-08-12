from __future__ import annotations

import abc
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ofen.common.onnx_utils import OnnxUtilities
from ofen.common.utils import auto_device
from ofen.configs.base.base_config import BaseConfig
from ofen.enums import ORTOptimizationLevel, Quantization
from ofen.models.base.model import BaseModel, ModelFeatures

if TYPE_CHECKING:
    from ofen.processors.base import BaseProcessor


@dataclasses.dataclass
class ORTModelConfig(BaseConfig):
    """Configuration for ONNX Runtime models."""

    name_or_path: str
    quantization: Quantization = Quantization.NONE
    optimization_level: ORTOptimizationLevel = ORTOptimizationLevel.O2
    simplify: bool = False
    enable_mem_reuse: bool = True
    enable_mem_pattern: bool = True

    @classmethod
    def from_pretrained(cls, name_or_path: str, **hf_kwargs: Any) -> ORTModelConfig:
        """Create a config from a pretrained model.

        Args:
        ----
            name_or_path (str): The name or path of the pretrained model.
            **hf_kwargs: Additional keyword arguments for Hugging Face's from_pretrained method.

        Returns:
        -------
            ORTModelConfig: The configuration for the ONNX Runtime model.

        """
        config = cls._from_registry(name_or_path)
        return config or cls(name_or_path=name_or_path)


class BaseORTModel(BaseModel):
    """Base class for ONNX Runtime models used for performing inference tasks.

    Attributes
    ----------
        config (ORTModelConfig): Configuration for the ONNX Runtime model.
        device (str): The device to run the model on.
        session: ONNX Runtime session for inference.

    """

    def __init__(
        self,
        name_or_path: str,
        *,
        processor: BaseProcessor,
        quantization: str | Quantization = Quantization.NONE,
        optimization_level: str | ORTOptimizationLevel = ORTOptimizationLevel.O2,
        simplify: bool = False,
        enable_mem_reuse: bool = True,
        enable_mem_pattern: bool = True,
        auto_export: bool = True,
        device: str = auto_device(),
    ) -> None:
        """Initialize the BaseORTModel.

        Args:
        ----
            name_or_path (str): Name or path of the model.
            processor (BaseProcessor): The processor to use for pre and post processing.
            quantization (Union[str, Quantization]): Quantization type for the model.
            optimization_level (Union[str, ORTOptimizationLevel]): Optimization level for ONNX Runtime.
            simplify (bool): Whether to simplify the ONNX model.
            enable_mem_reuse (bool): Whether to enable memory reuse in ONNX Runtime.
            enable_mem_pattern (bool): Whether to enable memory pattern optimization in ONNX Runtime.
            auto_export (bool): Whether to automatically export the model to ONNX if not found.
            device (str): The device to run the model on.

        """
        super().__init__(name_or_path=name_or_path, processor=processor)

        self.config = ORTModelConfig(
            name_or_path=name_or_path,
            quantization=Quantization(quantization),
            optimization_level=ORTOptimizationLevel(optimization_level),
            simplify=simplify,
            enable_mem_reuse=enable_mem_reuse,
            enable_mem_pattern=enable_mem_pattern,
        )
        self.device = device
        self.session = self._initialize_session(device, auto_export)

    @property
    @abc.abstractmethod
    def _auto_loading_cls(self) -> type:
        """Abstract property to define the auto-loading class for the model."""

    def _initialize_session(self, device: str, auto_export: bool = True):
        """Initialize the ONNX Runtime session.

        Args:
        ----
            device (str): The device to run the model on.
            auto_export (bool): Whether to automatically export the model to ONNX if not found.

        Returns:
        -------
            The initialized ONNX Runtime session.

        Raises:
        ------
            FileNotFoundError: If the ONNX file is not found and auto_export is False.

        """
        path = OnnxUtilities.get_onnx_path(
            name_or_path=self.config.name_or_path,
            quantization=self.config.quantization,
            simplified=self.config.simplify,
        )

        if Path(path).exists():
            return OnnxUtilities.get_onnx_session(path, device=self.device)

        original_path = OnnxUtilities.get_onnx_path(self.config.name_or_path)
        if Path(original_path).exists():
            optimized_path = OnnxUtilities.optimize(
                name_or_path=self.config.name_or_path,
                quantization=self.config.quantization,
                simplify=self.config.simplify,
            )
            return OnnxUtilities.get_onnx_session(optimized_path, device=self.device)

        if not auto_export:
            msg = f"ONNX file not found at {path}."
            raise FileNotFoundError(msg)

        model_cls = self._auto_loading_cls
        model = model_cls.from_pretrained(self.config.name_or_path)
        model.export_to_onnx()
        return self._initialize_session(device=device)

    def forward(self, **kwargs: dict[str, Any]) -> ModelFeatures:
        """Perform a forward pass through the model.

        Args:
        ----
            **kwargs: Input tensors for the model.

        Returns:
        -------
            ModelFeatures: Output features from the model.

        """
        return self.session.run(None, kwargs)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> BaseORTModel:
        """Create a BaseORTModel instance from a pretrained model.

        Args:
        ----
            model_name_or_path (str): Name or path of the pretrained model.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
        -------
            BaseORTModel: Instantiated BaseORTModel.

        """
        config = ORTModelConfig.from_pretrained(model_name_or_path)
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        return cls(**config_dict)
