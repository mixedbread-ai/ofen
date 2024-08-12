from __future__ import annotations

import dataclasses
import string
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers import AutoModel

from ofen.common.tensor_utils import TorchModule
from ofen.common.utils import auto_device
from ofen.configs.base.base_config import BaseConfig
from ofen.enums import NormalizationStrategy
from ofen.models.base.model import BaseModel
from ofen.models.torch.layers.normalization_layer import NormalizationLayer
from ofen.processors import TextProcessor
from ofen.processors.colbert_processor import ColbertProcessor

if TYPE_CHECKING:
    from ofen.types import ModelFeatures


@dataclasses.dataclass
class ColBERTOnnxExportInput:
    """Input structure for ONNX export of ColBERT model."""

    query: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]
    document: tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]


@dataclasses.dataclass
class ColBERTConfig(BaseConfig):
    """Configuration class for ColBERT model."""

    name_or_path: str
    output_dim: int = 128

    model_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    processor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> ColBERTConfig:
        """Create a ColBERTConfig instance from a pretrained model.

        Args:
        ----
            model_name (str): Name or path of the pretrained model.
            **kwargs: Additional keyword arguments for configuration.

        Returns:
        -------
            ColBERTConfig: An instance of ColBERTConfig.

        """
        return cls(name_or_path=model_name, **kwargs)


class ColBERT(TorchModule, BaseModel):
    """ColBERT (Contextualized Late Interaction over BERT) model implementation.

    This model uses a pretrained BERT-like model and applies a linear transformation
    to create contextualized embeddings for efficient retrieval.
    """

    def __init__(
        self,
        name_or_path: str,
        *,
        output_dim: int = 128,
        mask_punctuation: bool = True,
        device: str = auto_device(),
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        processor: TextProcessor | None = None,
    ) -> None:
        """Initialize the ColBERT model.

        Args:
        ----
            name_or_path (str): Name or path of the pretrained model.
            output_dim (int): Dimension of the output embeddings. Defaults to 128.
            mask_punctuation (bool): Whether to mask punctuation tokens. Defaults to True.
            model_kwargs (Dict[str, Any]): Additional keyword arguments for the pretrained model.
            device (str): Device to run the model on. Defaults to auto_device().
            train (bool): Whether to set the model in training mode. Defaults to False.

        """
        if model_kwargs is None:
            model_kwargs = {}
        if not processor:
            processor = ColbertProcessor.from_pretrained(name_or_path, **processor_kwargs)

        BaseModel.__init__(self, name_or_path=name_or_path, processor=processor)
        TorchModule.__init__(self)

        self.config = ColBERTConfig(name_or_path=name_or_path, output_dim=output_dim)
        self.device = device

        self.pretrained_model = AutoModel.from_pretrained(name_or_path, **model_kwargs)
        self._linear = nn.Linear(self.pretrained_model.config.hidden_size, output_dim, bias=False)
        self._normalization_layer = NormalizationLayer(strategy=NormalizationStrategy.L2)

        if mask_punctuation:
            tokenizer = TextProcessor.from_pretrained(name_or_path)
            self.register_buffer(
                "_skip_list",
                torch.tensor(
                    [tokenizer(symbol, add_special_tokens=False)["input_ids"][0] for symbol in string.punctuation]
                ),
            )
        else:
            self.register_buffer("_skip_list", None)

        self.to(self.device)

    def _get_punctuation_mask(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Generate a punctuation mask for the input tokens.

        Args:
        ----
            input_ids (torch.Tensor): Input token IDs.

        Returns:
        -------
            Optional[torch.Tensor]: A boolean mask where True indicates non-punctuation tokens.

        """
        if self._skip_list is None:
            return None
        return (input_ids.unsqueeze(1) != self._skip_list).all(dim=1)

    def _forward_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for queries.

        Args:
        ----
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
        -------
            torch.Tensor: Normalized query embeddings.

        """
        token_embeddings = self.pretrained_model(input_ids, attention_mask, return_dict=True).last_hidden_state
        token_embeddings = self._linear(token_embeddings)

        return self._normalization_layer(token_embeddings)

    def _forward_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for documents.

        Args:
        ----
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
        -------
            torch.Tensor: Normalized document embeddings.

        """
        token_embeddings = self.pretrained_model(input_ids, attention_mask, return_dict=True).last_hidden_state
        token_embeddings = self._linear(token_embeddings)

        punc_mask = self._get_punctuation_mask(input_ids) if self._skip_list else None
        if punc_mask is not None:
            token_embeddings = token_embeddings * punc_mask.unsqueeze(-1)

        return self._normalization_layer(token_embeddings)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_query: bool = False) -> ModelFeatures:
        """Forward pass of the ColBERT model.

        Args:
        ----
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            is_query (bool): Whether the input is a query. Defaults to False.

        Returns:
        -------
            ModelFeatures: A dictionary containing the 'embeddings' key with the computed embeddings.

        """
        with torch.inference_mode(mode=not self.training):
            embeddings = (
                self._forward_query(input_ids, attention_mask)
                if is_query
                else self._forward_document(input_ids, attention_mask)
            )

            return {"embeddings": embeddings}

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> ColBERT:
        """Create a ColBERT instance from a pretrained model.

        Args:
        ----
            name_or_path (str): Name or path of the pretrained model.
            **kwargs: Additional keyword arguments for model initialization.

        Returns:
        -------
            ColBERT: An instance of the ColBERT model.

        """
        config = ColBERTConfig.from_pretrained(name_or_path)
        cfg_dict = config.to_dict()
        cfg_dict.update(kwargs)
        return cls(name_or_path=name_or_path, **cfg_dict)
