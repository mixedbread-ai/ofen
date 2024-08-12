from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from typing import Any, Literal, Union

import numpy as np
from tokenizers import Encoding, Tokenizer

from ofen.common import hf_utils
from ofen.configs.base.base_config import BaseConfig
from ofen.exceptions import TruncationError
from ofen.processors.base import BaseProcessor
from ofen.types import ModelFeatures, ModelOutputs, Overflow

TextTokenizerInput = Union[str, list[str], list[int], list[list[int]]]
ReturnTensors = Literal["np", "pt"]


def to_torch(features: ModelFeatures) -> ModelFeatures:
    """Convert the features to PyTorch tensors.

    Args:
    ----
        features (ModelFeatures): The input features.

    Returns:
    -------
        ModelFeatures: The features as PyTorch tensors.

    """
    import torch
    return {k: torch.tensor(v) for k, v in features.items()}


@dataclasses.dataclass
class TextProcessorConfig(BaseConfig):
    """Configuration for TextProcessor."""

    name_or_path: str
    max_length: int
    cls_token: str = "[CLS]"
    pad_token: str = "[PAD]"
    sep_token: str = "[SEP]"
    unk_token: str = "[UNK]"
    mask_token: str = "[MASK]"
    truncation_side: Literal["left", "right"] = "right"
    tokenizer_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    return_token_type_ids: bool = False

    @classmethod
    def from_pretrained(cls, name_or_path: str, **hf_kwargs) -> BaseConfig:
        """Load configuration from a pretrained model.

        Args:
        ----
            name_or_path (str): The name or path of the pretrained model.
            **hf_kwargs: Additional keyword arguments for Hugging Face models.

        Returns:
        -------
            BaseConfig: The loaded configuration.

        Raises:
        ------
            ValueError: If the tokenizer config cannot be loaded.

        """
        config_path = (
            hf_utils.download_file_from_hf(name_or_path, filename="tokenizer_config.json", **hf_kwargs)
            or f"{name_or_path}/tokenizer_config.json"
        )
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            msg = f"Could not load tokenizer config for model {name_or_path}"
            raise ValueError(msg)

        return_token_type_ids = hf_utils.get_model_config(name_or_path, **hf_kwargs).get("type_vocab_size", 0) > 0

        def _get_token(x: str | dict[str, str]) -> str:
            return x["content"] if isinstance(x, dict) else x

        st_config = (
            hf_utils.get_sentence_transformer_config(name_or_path, **hf_kwargs)
            if hf_utils.is_sentence_transformer(name_or_path)
            else None
        )
        max_length = (
            st_config.max_seq_length if st_config else config.get("max_length") or config.get("model_max_length")
        )

        return cls(
            name_or_path=name_or_path,
            tokenizer_kwargs=hf_kwargs,
            max_length=max_length,
            truncation_side=config.get("truncation_side", "right"),
            cls_token=_get_token(config.get("cls_token")),
            pad_token=_get_token(config.get("pad_token")),
            sep_token=_get_token(config.get("sep_token")),
            unk_token=_get_token(config.get("unk_token")),
            mask_token=_get_token(config.get("mask_token")),
            return_token_type_ids=return_token_type_ids,
        )


class TextProcessor(BaseProcessor):
    """Tokenizer class for text processing using pre-trained models.

    Attributes
    ----------
        name_or_path (str): Name or path of the model.
        config (TextProcessorConfig): Configuration for the text processor.
        _tokenizer (Tokenizer): Tokenizer for encoding text sequences.
        return_tensors (ReturnTensors): Type of tensors to return.
        return_token_type_ids (bool): Whether to return token type IDs.
        cls_token_id (int): ID of the CLS token.
        pad_token_id (int): ID of the PAD token.
        sep_token_id (int): ID of the SEP token.
        unk_token_id (int): ID of the UNK token.
        mask_token_id (int): ID of the MASK token.

    """

    def __init__(
        self,
        name_or_path: str,
        *,
        max_length: int,
        truncation_side: Literal["left", "right"] = "right",
        tokenizer_kwargs: dict[str, Any] | None = None,
        cls_token: str = "[CLS]",
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        unk_token: str = "[UNK]",
        mask_token: str = "[MASK]",
        return_tensors: ReturnTensors = "pt",
        return_token_type_ids: bool = True,
    ):
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.name_or_path = name_or_path
        self.config = TextProcessorConfig(
            name_or_path=name_or_path,
            max_length=max_length,
            truncation_side=truncation_side,
            tokenizer_kwargs=tokenizer_kwargs,
            cls_token=cls_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            mask_token=mask_token,
        )

        self._tokenizer = Tokenizer.from_pretrained(name_or_path, **tokenizer_kwargs)
        self._tokenizer.enable_truncation(max_length=max_length, direction=self.config.truncation_side)
        self._tokenizer.no_padding()
        self.return_tensors = return_tensors
        self.return_token_type_ids = return_token_type_ids

        self.cls_token_id = self._tokenizer.token_to_id(self.config.cls_token)
        self.pad_token_id = self._tokenizer.token_to_id(self.config.pad_token)
        self.sep_token_id = self._tokenizer.token_to_id(self.config.sep_token)
        self.unk_token_id = self._tokenizer.token_to_id(self.config.unk_token)
        self.mask_token_id = self._tokenizer.token_to_id(self.config.mask_token)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> TextProcessor:
        """Load a pre-trained tokenizer from Hugging Face.

        Args:
        ----
            model_name_or_path (str): The name or path of the pre-trained model.
            **kwargs: Additional arguments for downloading the tokenizer.

        Returns:
        -------
            TextProcessor: An instance of the TextProcessor class.

        """
        text_processor_config = TextProcessorConfig.from_pretrained(model_name_or_path)
        cfg_dict = text_processor_config.to_dict()
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    @staticmethod
    def _sanitize_input(text: TextTokenizerInput, text_pair: TextTokenizerInput | None) -> list[str]:
        """Sanitize the input text and text pairs.

        Args:
        ----
            text (TextTokenizerInput): The input text.
            text_pair (Optional[TextTokenizerInput]): The input text pair.

        Returns:
        -------
            List[str]: Sanitized text and text pair.

        Raises:
        ------
            AssertionError: If input and pair have different lengths.

        """
        if isinstance(text, str):
            text = [text]
        if text_pair is not None:
            if isinstance(text_pair, str):
                text_pair = [text_pair]
            assert len(text) == len(text_pair), "Input and pair must have the same length"
            return list(zip(text, text_pair))
        return text

    def _convert_encoding(
        self,
        encoding: list[Encoding],
        return_token_type_ids: bool = False,
        return_attention_mask: bool = True,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        max_length: int | None = None,
        pad_token_id: int | None = None,
    ) -> tuple[ModelFeatures, bool]:
        """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        has_overflow = False
        encoding_dict = defaultdict(lambda: np.zeros((len(encoding), max_length), dtype=np.int32))
        encoding_dict["input_ids"] = np.full((len(encoding), max_length), pad_token_id, dtype=np.int64)

        max_seq_length = 0
        for i, enc in enumerate(encoding):
            max_seq_length = max(max_seq_length, len(enc.ids))
            encoding_dict["input_ids"][i, : len(enc.ids)] = enc.ids
            if return_token_type_ids:
                encoding_dict["token_type_ids"][i, : len(enc.type_ids)] = enc.type_ids
            if return_attention_mask:
                encoding_dict["attention_mask"][i, : len(enc.attention_mask)] = enc.attention_mask
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"][i, : len(enc.special_tokens_mask)] = enc.special_tokens_mask
            if return_length:
                length = len(enc.ids)
                if enc.overflowing:
                    has_overflow = True
                    length += sum(len(e.ids) for e in enc.overflowing)
                encoding_dict["length"][i] = length

        for key in encoding_dict:
            encoding_dict[key] = encoding_dict[key][:, :max_seq_length]

        return encoding_dict, has_overflow

    def _encode(
        self,
        *,
        text: TextTokenizerInput,
        pad_token_id: int | None = None,
        return_token_type_ids: bool | None = None,
        return_tensors: ReturnTensors | None = None,
        add_special_tokens: bool = True,
        raise_overflow_exception: bool = False,
        max_length: int | None = None,
    ) -> ModelFeatures:
        """Encode the input text and text pairs.

        Args:
        ----
            text (TextTokenizerInput): The input text.
            pad_token_id (Optional[int]): The ID of the padding token.
            return_token_type_ids (Optional[bool]): Whether to return token type IDs.
            return_tensors (Optional[ReturnTensors]): The return tensor type.
            add_special_tokens (bool): Whether to add special tokens.
            raise_overflow_exception (bool): Whether to raise an overflow exception.
            max_length (Optional[int]): The maximum length of the sequence.

        Returns:
        -------
            ModelFeatures: The encoded features.

        Raises:
        ------
            TruncationError: If truncation is not allowed and the input exceeds the maximum length.

        """
        max_length = max_length if max_length is not None else self.config.max_length
        return_tensors = return_tensors if return_tensors is not None else self.return_tensors
        return_token_type_ids = return_token_type_ids if return_token_type_ids is not None else self.return_token_type_ids
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id

        encodings = self._tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)
        features, has_overflow = self._convert_encoding(
            encodings,
            pad_token_id=pad_token_id,
            return_token_type_ids=return_token_type_ids,
            max_length=max_length,
            return_length=raise_overflow_exception,
        )

        if raise_overflow_exception and has_overflow:
            overflowing_tokens = [(i, length) for i, length in enumerate(features["length"]) if length > max_length]
            raise TruncationError(
                name_or_path=self.name_or_path,
                max_length=self.config.max_length,
                overflows=[Overflow(index=i, length=length) for i, length in overflowing_tokens],
            )

        return to_torch(features) if return_tensors == "pt" else features

    def __call__(
        self,
        text: TextTokenizerInput,
        text_pair: TextTokenizerInput | None = None,
        *,
        pad_token_id: int | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        return_tensors: ReturnTensors | None = None,
        return_token_type_ids: bool | None = None,
        raise_overflow_exception: bool = False,
    ) -> ModelFeatures:
        """Call method for encoding text and text pairs.

        Args:
        ----
            text (TextTokenizerInput): The input text.
            text_pair (Optional[TextTokenizerInput]): The input text pair.
            pad_token_id (Optional[int]): The ID of the padding token.
            max_length (Optional[int]): The maximum length of the sequence.
            return_tensors (Optional[ReturnTensors]): The return tensor type.
            return_token_type_ids (bool): Whether to return token type IDs.
            raise_overflow_exception (bool): Whether to raise an overflow exception.

        Returns:
        -------
            ModelFeatures: The encoded features.

        """
        return self._encode(
            text=self._sanitize_input(text, text_pair),
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            return_token_type_ids=return_token_type_ids,
            raise_overflow_exception=raise_overflow_exception,
            pad_token_id=pad_token_id,
            max_length=max_length,
        )

    def pre_process(self, *args, **kwargs: Any) -> ModelFeatures:
        """Pre-process the input data.

        Args:
        ----
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
        -------
            ModelFeatures: The pre-processed features.

        """
        return self(*args, **kwargs)

    def post_process(self, _: ModelFeatures, outputs: ModelOutputs) -> ModelOutputs:
        """Post-process the model outputs.

        Args:
        ----
            _ (ModelFeatures): Unused input features.
            outputs (ModelOutputs): The model outputs.

        Returns:
        -------
            ModelOutputs: The post-processed outputs.

        """
        return outputs
