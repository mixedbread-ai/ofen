from __future__ import annotations

import dataclasses
from typing import Any, overload

from transformers import AutoProcessor

from ofen.common.tensor_utils import TorchModule
from ofen.common.utils import auto_device, ensure_import
from ofen.enums import NormalizationStrategy, PoolingStrategy
from ofen.models.base.encoder import EncoderOutput
from ofen.models.base.model import BaseModel
from ofen.models.mixins.onnx_exportable import OnnxExportable
from ofen.models.torch.layers.normalization_layer import NormalizationLayer
from ofen.models.torch.layers.pooling_layer import PoolingLayer
from ofen.pretrainable import Pretrainable
from ofen.processors import TextProcessor

with ensure_import("ofen[torch]"):
    import torch
    from transformers import CLIPModel


@dataclasses.dataclass
class ClipEncoderConfig(Pretrainable):
    """Configuration class for ClipEncoder.

    Attributes
    ----------
        name_or_path (str): Name or path of the pre-trained model.
        model_kwargs (dict): Keyword arguments for the model.
        text_processor_kwargs (dict): Keyword arguments for the text processor.
        image_processor_kwargs (dict): Keyword arguments for the image processor.
        pooling_strategy (PoolingStrategy): Strategy for pooling encoder outputs.
        normalize (bool): Whether to normalize the output embeddings.

    """

    name_or_path: str
    model_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    text_processor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    image_processor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    normalize: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_pretrained(cls, name_or_path: str, **hf_kwargs) -> ClipEncoderConfig:
        """Create a ClipEncoderConfig instance from a pre-trained model."""
        return cls(name_or_path=name_or_path, **hf_kwargs)


class ClipEncoder(TorchModule, OnnxExportable):
    """ClipEncoder for encoding text and image inputs into vectors.

    Attributes
    ----------
        config (ClipEncoderConfig): Configuration for the CLIP encoder.
        clip_model (CLIPModel): The underlying pre-trained CLIP model.
        text_model (CLIPTextModel): The text encoder part of CLIP.
        vision_model (CLIPVisionModel): The vision encoder part of CLIP.
        _pooling_layer (PoolingLayer): Layer for pooling encoder outputs.
        _normalization_layer (NormalizationLayer): Layer for normalizing embeddings.

    """

    def __init__(
        self,
        name_or_path: str,
        *,
        pooling_strategy: PoolingStrategy | str = PoolingStrategy.MEAN,
        normalize: bool = False,
        device: str = auto_device(),
        model_kwargs: dict[str, Any] | None = None,
        text_processor_kwargs: dict[str, Any] | None = None,
        image_processor_kwargs: dict[str, Any] | None = None,
        text_processor: TextProcessor | None = None,
        image_processor=None,
    ) -> None:
        """Initialize the ClipEncoder."""
        if model_kwargs is None:
            model_kwargs = {}
        if text_processor_kwargs is None:
            text_processor_kwargs = {}
        if image_processor_kwargs is None:
            image_processor_kwargs = {}

        TorchModule.__init__(self)
        BaseModel.__init__(self, name_or_path=name_or_path, processor=None)

        self.clip_model = CLIPModel.from_pretrained(name_or_path, **model_kwargs)

        if text_processor is None:
            self.text_processor = TextProcessor.from_pretrained(
                name_or_path, return_tensors="pt", **text_processor_kwargs
            )
        else:
            self.text_processor = text_processor

        if image_processor is None:
            self.image_processor = AutoProcessor.from_pretrained(name_or_path, **image_processor_kwargs)
        else:
            self.image_processor = image_processor

        self.config = ClipEncoderConfig(
            name_or_path=name_or_path,
            pooling_strategy=PoolingStrategy(pooling_strategy),
            normalize=normalize,
            model_kwargs=model_kwargs,
            text_processor_kwargs=text_processor_kwargs,
            image_processor_kwargs=image_processor_kwargs,
        )

        self._pooling_layer = PoolingLayer(strategy=self.config.pooling_strategy)
        self._normalization_layer = NormalizationLayer(
            strategy=NormalizationStrategy.L2 if normalize else NormalizationStrategy.NONE
        )

        self.to(device)
        self.eval()

    @overload
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutput: ...

    @overload
    def forward(self, pixel_values: torch.Tensor) -> EncoderOutput: ...

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Forward pass of the ClipEncoder."""
        if input_ids is None and pixel_values is None:
            msg = "Either text input (input_ids and attention_mask) or image input (pixel_values) must be provided."
            raise ValueError(msg)

        with torch.inference_mode(mode=not self.training):
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

            if pixel_values and input_ids:
                result = self.clip_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                text_embeddings = result.text_embeds
                image_embeddings = result.image_embeds
            elif pixel_values:
                image_embeddings = self.clip_model.get_image_features(pixel_values=pixel_values)
            else:
                text_embeddings = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

            if image_embeddings:
                image_embeddings = self._normalization_layer(image_embeddings)
            if text_embeddings:
                text_embeddings = self._normalization_layer(text_embeddings)

            return EncoderOutput(embeddings=embeddings.detach().cpu())

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> ClipEncoder:
        """Create a ClipEncoder instance from a pre-trained model."""
        return cls.from_config(ClipEncoderConfig.from_pretrained(name_or_path), **kwargs)

    @classmethod
    def from_config(cls, config: ClipEncoderConfig, **kwargs) -> ClipEncoder:
        """Create a ClipEncoder instance from a configuration object."""
        cfg_dict = config.to_dict()
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    def export_to_onnx(self, path: str | None = None) -> tuple[str, str]:
        """Export the model to ONNX format for both text and image inputs."""
        path = path or self.config.name_or_path
        text_path = f"{path}-text" if path else None
        image_path = f"{path}-image" if path else None

        text_onnx_path = self._export_to_onnx(["Hello world", "What is poppin"], text_path)
        image_onnx_path = self._export_to_onnx(torch.randn(1, 3, 224, 224), image_path)

        return text_onnx_path, image_onnx_path
