from __future__ import annotations

import abc
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from ofen.common.numpy_utils import (
    cosine_similarity,
    euclidean_distance,
    hamming_distance,
    inner_product,
    jensen_shannon_distance,
)
from ofen.common.tensor_utils import l2_normalize, quantize_embeddings
from ofen.common.utils import bucket_batch_iter, threaded_generator, tqdm
from ofen.enums import EncodingFormat
from ofen.models.base.model import BaseModel

if TYPE_CHECKING:
    from collections.abc import Generator

    from ofen.types import ModelFeatures

# Type aliases for better readability
MultiEncodingFormatReturn = dict[EncodingFormat, np.ndarray]


class EncoderOutput(TypedDict):
    """Output type for encoder models."""

    embeddings: np.ndarray


@dataclass
class EncodingResult:
    embeddings: np.ndarray | MultiEncodingFormatReturn
    total_tokens: int


@dataclass
class StreamedEncodingItem:
    embeddings: np.ndarray | MultiEncodingFormatReturn
    indices: list[int]
    tokens: int


class BaseEncoder(BaseModel):
    """Base class for encoder models."""

    @abc.abstractmethod
    def forward(self, **features: ModelFeatures) -> EncoderOutput:
        """Abstract method to be implemented by subclasses.

        Args:
        ----
            **features: Model features.

        Returns:
        -------
            EncoderOutput: The output of the encoder.

        """

    @abc.abstractmethod
    def encode(self, data: Any, **kwargs) -> EncodingResult:
        """Abstract method to encode data into embeddings."""

    def _preprocess_batch(self, batch: tuple[list[Any], list[int]], **kwargs) -> tuple[ModelFeatures, list[int]]:
        """Preprocess a batch of data."""
        return self.pre_process(batch[0], **kwargs), batch[1]

    def _encode(
        self,
        data: list[Any],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
        **kwargs,
    ) -> EncodingResult:
        """Internal method to encode data into embeddings.

        Args:
        ----
            data: Input data to encode.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep in the output embeddings.
            encoding_format: Format(s) for quantization.
            **kwargs: Additional keyword arguments for pre-processing.

        Returns:
        -------
            EncodingResult containing embeddings and total tokens.

        Raises:
        ------
            ValueError: If input data is empty.
        """
        if not data:
            msg = "Data cannot be empty"
            raise ValueError(msg)

        embeddings, total_tokens = self._compute_embeddings(
            data, batch_size=batch_size, show_progress=show_progress, **kwargs
        )

        embeddings = BaseEncoder._postprocess_embeddings(
            embeddings, normalize=normalize, dimensions=dimensions, encoding_format=encoding_format
        )

        return EncodingResult(embeddings=embeddings, total_tokens=total_tokens)

    def _compute_embeddings(
        self, data: list[Any], batch_size: int = 32, show_progress: bool = False, **kwargs
    ) -> tuple[np.ndarray, int]:
        """Compute embeddings for the given data.

        Args:
        ----
            data: List of data to encode.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            **kwargs: Additional keyword arguments for pre-processing.

        Returns:
        -------
            Tuple of computed embeddings and total token count.
        """
        if len(data) <= batch_size:
            features = self.pre_process(data, **kwargs)
            embeddings = np.array(self.forward(**features)["embeddings"])
            total_tokens = features["input_ids"].shape[0] * features["input_ids"].shape[1]
            return embeddings, total_tokens

        total_tokens = 0
        embeddings = np.zeros((len(data), self.embedding_dim), dtype=np.float32)
        for features, org_indices in tqdm(
            threaded_generator(
                functools.partial(self._preprocess_batch, **kwargs), bucket_batch_iter(data, batch_size)
            ),
            desc="Encoding",
            total=(len(data) + batch_size - 1) // batch_size,
            disable=not show_progress,
        ):
            embeddings[org_indices] = np.array(self.forward(**features)["embeddings"])
            total_tokens += features["input_ids"].shape[0] * features["input_ids"].shape[1]

        return embeddings, total_tokens

    def _stream_encode(
        self,
        data: list[Any],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
        **kwargs,
    ) -> Generator[StreamedEncodingItem, None, None]:
        """Internal method to stream embeddings for data.

        Args:
        ----
            data: Input data to encode.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep in the output embeddings.
            encoding_format: Format(s) for quantization.
            **kwargs: Additional keyword arguments for pre-processing.

        Yields:
        ------
            StreamedEncodingItem containing generated embeddings and indices for each batch.

        Raises:
        ------
            ValueError: If input data is empty.
        """
        if not data:
            msg = "Data cannot be empty"
            raise ValueError(msg)

        for features, prev_indices in tqdm(
            threaded_generator(
                functools.partial(self._preprocess_batch, **kwargs), bucket_batch_iter(data, batch_size)
            ),
            desc="Encoding",
            total=(len(data) + batch_size - 1) // batch_size,
            disable=not show_progress,
        ):
            tokens = features["input_ids"].shape[0] * features["input_ids"].shape[1]
            embeddings = np.array(self.forward(**features)["embeddings"])
            embeddings = BaseEncoder._postprocess_embeddings(
                embeddings, normalize=normalize, dimensions=dimensions, encoding_format=encoding_format
            )

            yield StreamedEncodingItem(embeddings=embeddings, indices=prev_indices, tokens=tokens)

    @staticmethod
    def _postprocess_embeddings(
        embeddings: np.ndarray,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
    ) -> np.ndarray | MultiEncodingFormatReturn:
        """Process embeddings by normalizing, truncating dimensions, and quantizing.

        Args:
        ----
            embeddings: Input embeddings.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep (if specified).
            encoding_format: Format(s) for quantization.

        Returns:
        -------
            Processed embeddings in the specified format(s).

        Raises:
        ------
            ValueError: If dimensions is negative.

        """
        embeddings = BaseEncoder.normalize_embeddings(embeddings) if normalize else embeddings
        embeddings = BaseEncoder.truncate_embeddings(embeddings, dimensions) if dimensions is not None else embeddings
        return (
            BaseEncoder.quantize_embeddings(embeddings, encoding_format) if encoding_format is not None else embeddings
        )

    @functools.cached_property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        res = self.forward(**self.pre_process(["hello world"]))
        return res["embeddings"].shape[1]

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings using L2 normalization."""
        return l2_normalize(embeddings)

    @staticmethod
    def truncate_embeddings(embeddings: np.ndarray, dimensions: int) -> np.ndarray:
        """Truncate embeddings to specified number of dimensions."""
        if dimensions <= 0:
            msg = "Dimensions must be a positive integer."
            raise ValueError(msg)
        return embeddings[:, :dimensions]

    @staticmethod
    def quantize_embeddings(
        embeddings: np.ndarray, encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str]
    ) -> np.ndarray | MultiEncodingFormatReturn:
        """Quantizes embeddings in specified format(s)."""
        if isinstance(encoding_format, (str, EncodingFormat)):
            return quantize_embeddings(embeddings, encoding_format=encoding_format, ranges=None)

        return {fmt: quantize_embeddings(embeddings, encoding_format=fmt, ranges=None) for fmt in encoding_format}

    @staticmethod
    def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the inner product between two sets of embeddings."""
        return inner_product(a, b)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the cosine similarity between two sets of embeddings."""
        return cosine_similarity(a, b)

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Euclidean distance between two sets of embeddings."""
        return euclidean_distance(a, b)

    @staticmethod
    def jensen_shannon_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Jensen-Shannon distance between two sets of embeddings."""
        return jensen_shannon_distance(a, b)

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Hamming distance between two sets of embeddings."""
        return hamming_distance(a, b)
