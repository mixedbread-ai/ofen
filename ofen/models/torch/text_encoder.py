from __future__ import annotations

import dataclasses
from typing import Any

from ofen.common.hf_utils import get_sentence_transformer_config, is_sentence_transformer
from ofen.common.torch_utils import TorchModule
from ofen.common.utils import auto_device, ensure_import
from ofen.enums import NormalizationStrategy, PoolingStrategy
from ofen.models.base.model import BaseModel
from ofen.models.base.text_encoder import BaseTextEncoder
from ofen.models.mixins.onnx_exportable import OnnxExportable
from ofen.models.torch.layers.normalization_layer import NormalizationLayer
from ofen.models.torch.layers.pooling_layer import PoolingLayer
from ofen.pretrainable import Pretrainable
from ofen.processors import TextProcessor
from ofen.types import NDArrayOrTensor

with ensure_import("ofen[torch]"):
    import torch
    from transformers import AutoModel


@dataclasses.dataclass
class TextEncoderConfig(Pretrainable):
    """Configuration class for TextEncoder.

    Attributes
        name_or_path (str): Name or path of the pre-trained model.
        model_kwargs (HFModelKwargs): Keyword arguments for the model.
        processor_kwargs (HFTokenizerKwargs): Keyword arguments for the processor.
        pooling_strategy (PoolingStrategy): Strategy for pooling encoder outputs.
        normalize (bool): Whether to normalize the output embeddings.

    """

    name_or_path: str
    model_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    processor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN
    normalize: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def get_pooling_strategy(name_or_path: str) -> PoolingStrategy:
        if is_sentence_transformer(name_or_path):
            sbert_config = get_sentence_transformer_config(name_or_path)
            return sbert_config.pooling_strategy
        return PoolingStrategy.NONE


class TextEncoder(BaseTextEncoder, TorchModule, OnnxExportable):
    """TextEncoder for encoding text sequences into vectors.

    Attributes
        config (TextEncoderConfig): Configuration for the text encoder.
        pretrained_model (AutoModel): The underlying pre-trained model.
        _pooling_layer (PoolingLayer): Layer for pooling encoder outputs.
        _normalization_layer (NormalizationLayer): Layer for normalizing embeddings.

    """

    def __init__(
        self,
        name_or_path: str,
        *,
        pooling_strategy: PoolingStrategy | str | None = None,
        normalize: bool = False,
        device: str = auto_device(),
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        processor: TextProcessor | None = None,
    ) -> None:
        """Initialize the TextEncoder.

        Args:
            name_or_path (str): Name or path of the pre-trained model.
            pooling_strategy (Union[PoolingStrategy, str]): Strategy for pooling encoder outputs.
            normalize (bool): Whether to normalize the output embeddings.
            device (str): Device to run the model on.
            model_kwargs (Dict[str, Any]): Additional keyword arguments for the model.
            processor_kwargs (Dict[str, Any]): Additional keyword arguments for the processor.
            processor (Optional[TextProcessor]): Custom text processor.

        Example:
        ```python
        encoder = TextEncoder.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
        ```

        """
        if processor_kwargs is None:
            processor_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if processor is None:
            processor = TextProcessor.from_pretrained(name_or_path, return_tensors="pt", **processor_kwargs)

        TorchModule.__init__(self)
        BaseModel.__init__(self, name_or_path=name_or_path, processor=processor)

        self.pretrained_model = AutoModel.from_pretrained(name_or_path, **model_kwargs)
        self.config = TextEncoderConfig(
            name_or_path=name_or_path,
            pooling_strategy=PoolingStrategy(pooling_strategy)
            if pooling_strategy is not None
            else TextEncoderConfig.get_pooling_strategy(name_or_path),
            normalize=normalize,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
        )

        self._warn_if_invalid_params()

        self._pooling_layer = PoolingLayer(strategy=self.config.pooling_strategy)
        self._normalization_layer = NormalizationLayer(
            strategy=NormalizationStrategy.L2 if normalize else NormalizationStrategy.NONE
        )

        self.to(device)
        self.eval()

    def _warn_if_invalid_params(self) -> None:
        """Warn if pooling strategy is set to NONE."""
        if self.config.pooling_strategy == PoolingStrategy.NONE:
            import warnings

            warnings.warn(
                "Pooling strategy is set to NONE. This may result in unexpected behavior "
                "as the output will be the full sequence of hidden states rather than a "
                "single vector per input. Consider using a different pooling strategy "
                "such as mean, max, or cls if you need a fixed-size representation.",
                UserWarning,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        convert_to_numpy: bool = True,
    ) -> dict[str, NDArrayOrTensor]:
        """Forward pass of the TextEncoder.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (Optional[torch.Tensor]): The token type IDs.

        Returns:
            EncoderOutput: An object containing the embeddings.

        """
        with torch.inference_mode(mode=not self.training):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            outputs = self.pretrained_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True
            )
            embeddings = self._pooling_layer(
                attention_mask=attention_mask, last_hidden_state=outputs["last_hidden_state"]
            )
            embeddings = self._normalization_layer(embeddings)

            return {"embeddings": embeddings.cpu() if convert_to_numpy else embeddings}

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> TextEncoder:
        """Create a TextEncoder instance from a pre-trained model.

        Args:
            name_or_path (str): Name or path of the pre-trained model.
            **kwargs: Additional keyword arguments.

        Returns:
            TextEncoder: An instance of TextEncoder.

        """
        return cls.from_config(TextEncoderConfig.from_pretrained(name_or_path), **kwargs)

    @classmethod
    def from_config(cls, config: TextEncoderConfig, **kwargs) -> TextEncoder:
        """Create a TextEncoder instance from a configuration object.

        Args:
            config (TextEncoderConfig): Configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            TextEncoder: An instance of TextEncoder.

        """
        cfg_dict = config.to_dict()
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    def export_to_onnx(self, path: str | None = None) -> str:
        """Export the model to ONNX format.

        Args:
            path (Optional[str]): Path to save the ONNX model.

        Returns:
            str: Path to the exported ONNX model.

        """
        return self._export_to_onnx(["Hello world", "What is poppin"], path)
