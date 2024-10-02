from __future__ import annotations

import abc
import functools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
from ofen.types import NDArrayOrTensor

if TYPE_CHECKING:
    from ofen.types import ModelFeatures


# Type aliases for better readability
MultiEncodingFormatReturn = dict[EncodingFormat, NDArrayOrTensor]


@dataclass
class EncodingResult:
    outputs: dict[str, NDArrayOrTensor | MultiEncodingFormatReturn]
    total_tokens: int

    def __getitem__(self, key):
        return self.outputs[key]

    def __contains__(self, key):
        return key in self.outputs

    def get(self, key, default=None):
        return self.outputs.get(key, default)


class BaseEncoder(BaseModel):
    """Base class for encoder models."""

    @abc.abstractmethod
    def forward(self, **features: ModelFeatures) -> dict[str, NDArrayOrTensor]:
        """Abstract method to be implemented by subclasses.

        Args:
            **features: Model features.

        Returns:
            Dict[str, Union[np.ndarray, torch.Tensor]]: The outputs of the encoder.

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
        *,
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
        return_numpy: bool = True,
        **kwargs,
    ) -> EncodingResult:
        """Internal method to encode data into embeddings.

        Args:
            data: Input data to encode.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep in the output embeddings.
            encoding_format: Format(s) for quantization.
            **kwargs: Additional keyword arguments for pre-processing.

        Returns:
            EncodingResult containing outputs and total tokens.

        Raises:
            ValueError: If input data is empty.
        """
        if not data:
            msg = "Data cannot be empty"
            raise ValueError(msg)

        if not isinstance(data, list):
            data = [data]

        outputs, total_tokens = self._compute_embeddings(
            data,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=normalize,
            dimensions=dimensions,
            encoding_format=encoding_format,
            return_numpy=return_numpy,
            **kwargs,
        )

        return EncodingResult(outputs=outputs, total_tokens=total_tokens)

    def _compute_embeddings(
        self,
        data: list[Any],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
        return_numpy: bool = True,
        **kwargs,
    ) -> tuple[dict[str, NDArrayOrTensor | MultiEncodingFormatReturn], int]:
        """Compute embeddings for the given data.

        Args:
            data: List of data to encode.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep in the output embeddings.
            encoding_format: Format(s) for quantization.
            output_kind: Desired output kind ("numpy" or "torch").
            **kwargs: Additional keyword arguments for pre-processing.

        Returns:
            Tuple of computed outputs and total token count.
        """
        post_processing_fn = functools.partial(
            self._postprocess_embedding,
            normalize=normalize,
            dimensions=dimensions,
            encoding_format=encoding_format,
            return_numpy=return_numpy,
        )

        if len(data) <= batch_size:
            features = self.pre_process(data, **kwargs)
            total_tokens = math.prod(features["input_ids"].shape)

            outputs = self.forward(**features)
            return {key: post_processing_fn(output) for key, output in outputs.items()}, total_tokens

        total_tokens = 0
        outputs_dict = defaultdict(lambda: [None] * len(data))
        for features, org_indices in tqdm(
            threaded_generator(
                functools.partial(self._preprocess_batch, **kwargs), bucket_batch_iter(data, batch_size)
            ),
            desc="Encoding",
            total=(len(data) + batch_size - 1) // batch_size,
            disable=not show_progress,
        ):
            outputs = self.forward(**features)
            total_tokens += math.prod(features["input_ids"].shape)

            for key, output in outputs.items():
                outputs_dict[key][org_indices] = post_processing_fn(output)

        return outputs_dict, total_tokens

    def _postprocess_embedding(
        self,
        embeddings: NDArrayOrTensor,
        *,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
        return_numpy: bool = True,
    ) -> NDArrayOrTensor:
        """Process a single embedding by normalizing, truncating dimensions, and quantizing.

        Args:
            embeddings: Input embeddings.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep (if specified).
            encoding_format: Format(s) for quantization.
            return_numpy: Whether to return a numpy array.
        Returns:
            Processed embeddings in the specified format(s).
        """
        if normalize:
            embeddings = self.normalize_embeddings(embeddings)
        if dimensions is not None:
            embeddings = self.truncate_embeddings(embeddings, dimensions)

        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.cpu()

            if return_numpy:
                embeddings = embeddings.numpy()

        if encoding_format is not None:
            embeddings = embeddings if isinstance(embeddings, np.ndarray) else embeddings.numpy()
            embeddings = self.quantize_embeddings(embeddings, encoding_format)

            if not return_numpy:
                import torch

                embeddings = (
                    {fmt: torch.from_numpy(emb) for fmt, emb in embeddings.items()}
                    if isinstance(embeddings, dict)
                    else torch.from_numpy(embeddings)
                )
        return embeddings

    @staticmethod
    def quantize_embeddings(
        embeddings: NDArrayOrTensor,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str],
    ) -> np.ndarray | MultiEncodingFormatReturn:
        """Quantizes embeddings in specified format(s)."""
        if isinstance(encoding_format, (str, EncodingFormat)):
            return quantize_embeddings(embeddings, encoding_format=encoding_format)

        return {fmt: quantize_embeddings(embeddings, encoding_format=fmt) for fmt in encoding_format}

    @staticmethod
    def normalize_embeddings(embeddings: NDArrayOrTensor) -> NDArrayOrTensor:
        """Normalize embeddings using L2 normalization."""
        return l2_normalize(embeddings)

    @staticmethod
    def truncate_embeddings(embeddings: NDArrayOrTensor, dimensions: int) -> NDArrayOrTensor:
        """Truncate embeddings to specified number of dimensions."""
        return embeddings[..., :dimensions]

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
