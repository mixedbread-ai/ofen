from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForSequenceClassification

from ofen.common.tensor_utils import TorchModule
from ofen.common.utils import auto_device
from ofen.configs.base.base_config import BaseConfig
from ofen.enums import ActivationStrategy
from ofen.models.base.cross_encoder import BaseCrossEncoder, CrossEncoderOutput
from ofen.models.mixins.onnx_exportable import OnnxExportable
from ofen.models.torch.layers import ActivationLayer
from ofen.pretrainable import Pretrainable
from ofen.processors import TextProcessor


@dataclasses.dataclass
class CrossEncoderConfig(BaseConfig, Pretrainable):
    """Configuration class for CrossEncoder."""

    name_or_path: str
    activation_strategy: ActivationStrategy = ActivationStrategy.SIGMOID
    model_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> CrossEncoderConfig:
        """Create a CrossEncoderConfig instance from a pre-trained model.

        Args:
        ----
            model_name_or_path (str): Name or path of the pre-trained model.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            CrossEncoderConfig: An instance of CrossEncoderConfig.

        """
        config = cls._from_registry(model_name_or_path)
        if config:
            return config
        return cls(name_or_path=model_name_or_path, **kwargs)


class CrossEncoder(BaseCrossEncoder, TorchModule, OnnxExportable):
    """CrossEncoder for performing cross-encoding tasks such as sentence-pair classification and ranking.

    Attributes
    ----------
        config (CrossEncoderConfig): Configuration for the cross encoder.
        _pretrained_model (AutoModelForSequenceClassification): The underlying pre-trained model.
        _activation_layer (ActivationLayer): Layer for activation function.

    """

    def __init__(
        self,
        name_or_path: str,
        *,
        activation_strategy: str | ActivationStrategy = ActivationStrategy.SIGMOID,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        device: str = auto_device(),
        processor: TextProcessor | None = None,
    ) -> None:
        """Initialize the CrossEncoder.

        Args:
        ----
            name_or_path (str): Name or path of the pre-trained model.
            activation_strategy (Union[str, ActivationStrategy]): Strategy for activation function.
            model_kwargs (Dict[str, Any]): Additional keyword arguments for the model.
            processor_kwargs (Dict[str, Any]): Additional keyword arguments for the processor.
            device (str): Device to run the model on.
            processor (Optional[TextProcessor]): Custom text processor.

        """
        if processor_kwargs is None:
            processor_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if not processor:
            processor = TextProcessor.from_pretrained(name_or_path, **processor_kwargs, return_tensors="pt")

        BaseCrossEncoder.__init__(self, name_or_path=name_or_path, processor=processor)
        TorchModule.__init__(self)

        self.config = CrossEncoderConfig(
            name_or_path=name_or_path,
            model_kwargs=model_kwargs,
            activation_strategy=activation_strategy,
        )
        self._pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_kwargs.get("name_or_path") or name_or_path, **model_kwargs
        )
        self._activation_layer = ActivationLayer(strategy=activation_strategy)

        self.eval()
        self.to(device)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> CrossEncoderOutput:
        """Forward pass of the CrossEncoder.

        Args:
            input_ids (Tensor): The input token IDs.
            attention_mask (Tensor): The attention mask.

        Returns:
            CrossEncoderOutput: An object containing the scores.

        """
        with torch.inference_mode(not self.training):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            logits = self._pretrained_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)[
                "logits"
            ]
            scores = self._activation_layer(logits=logits)
        return CrossEncoderOutput(scores=scores.detach().cpu())

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> CrossEncoder:
        """Create a CrossEncoder instance from a pre-trained model.

        Args:
        ----
            model_name_or_path (str): Name or path of the pre-trained model.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            CrossEncoder: An instance of CrossEncoder.

        """
        return cls.from_config(CrossEncoderConfig.from_pretrained(model_name_or_path), **kwargs)

    @classmethod
    def from_config(cls, config: CrossEncoderConfig, **kwargs: Any) -> CrossEncoder:
        """Create a CrossEncoder instance from a configuration object.

        Args:
        ----
            config (CrossEncoderConfig): Configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            CrossEncoder: An instance of CrossEncoder.

        """
        cfg_dict = config.to_dict()
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    def export_to_onnx(self, path: str | None = None) -> str:
        """Export the model to ONNX format.

        Args:
        ----
            path (Optional[str]): Path to save the ONNX model.

        Returns:
        -------
            str: Path to the exported ONNX model.

        """
        return self._export_to_onnx(("What is love?", "Baby don't hurt me."), path)
