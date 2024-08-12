from __future__ import annotations

import abc
from dataclasses import dataclass
import dataclasses
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from tqdm import tqdm

from ofen.common import numpy_utils
from ofen.common.utils import bucket_batch_iter
from ofen.models.base.model import BaseModel
from ofen.runners.runner import Runner
from ofen.types import ModelFeatures

if TYPE_CHECKING:
    from torch import Tensor


class CrossEncoderOutput(TypedDict):
    """Output type for cross encoder models."""

    scores: Tensor | np.ndarray

@dataclasses.dataclass
class RankResult:
    """Represents the result of a ranking operation."""
    score: float
    index: int
    input: str | dict | None = None

@dataclass
class CrossEncoderResult:
    """Output of cross-encoder ranking."""
    results: list[RankResult]
    total_tokens: int

    @property
    def inputs(self) -> list[str]:
        """Get the input documents."""
        return [result.input for result in self.results]

    @property
    def scores(self) -> np.ndarray:
        """Get the scores."""
        return np.array([result.score for result in self.results])

    @property
    def indices(self) -> np.ndarray:
        """Get the indices."""
        return np.array([result.index for result in self.results])

class BaseCrossEncoder(BaseModel):
    """Base class for cross encoder models."""

    @abc.abstractmethod
    def forward(self, **features: ModelFeatures) -> CrossEncoderOutput:
        """Abstract method to be implemented by subclasses.

        Args:
        ----
            **features: Model features.

        Returns:
        -------
            CrossEncoderOutput: The output of the cross encoder.

        """

    @Runner.with_metrics
    def rerank(
        self, query: str, documents: list[str], *, top_k: int = 100, sort: bool = True, batch_size: int = 32, return_input: bool = False, **kwargs
    ) -> CrossEncoderResult:
        """Rerank documents based on a single query.

        Args:
        ----
            query: The query string.
            documents: List of document strings to rerank.
            top_k: Number of top results to return. Defaults to 100.
            sort: Whether to sort the results. Defaults to True.
            batch_size: Batch size for processing. Defaults to 32.
            return_input: Whether to include input documents in the result. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            CrossEncoderResult: Reranked results.

        """
        return self.rank(
            queries=[query] * len(documents),
            documents=documents,
            top_k=top_k,
            sort=sort,
            batch_size=batch_size,
            return_input=return_input,
            **kwargs,
        )

    def rank(
        self,
        queries: list[str],
        documents: list[str],
        *,
        top_k: int = 100,
        sort: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
        return_input: bool = False,
        **kwargs,
    ) -> CrossEncoderResult:
        """Rank documents based on queries.

        Args:
            queries: List of query strings.
            documents: List of document strings to rank.
            top_k: Number of top results to return. Defaults to 100.
            sort: Whether to sort the results. Defaults to True.
            batch_size: Batch size for processing. Defaults to 32.
            show_progress: Whether to show progress bar. Defaults to False.
            return_input: Whether to include input documents in the result. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            CrossEncoderResult: Ranked results including rerank results, optionally input documents, and token count.

        Raises:
            ValueError: If the number of queries and documents are not equal.

        """
        if len(queries) != len(documents):
            msg = "The number of queries and documents must be the same."
            raise ValueError(msg)

        scores, total_tokens = self._compute_scores(queries, documents, batch_size, show_progress, **kwargs)

        top_k_scores, top_k_indices = numpy_utils.top_k_numpy(scores=scores, k=top_k, sort=sort)

        results = [RankResult(index=i, score=score, input=documents[i] if return_input else None) for i, score in zip(top_k_indices, top_k_scores)]

        return CrossEncoderResult(
            total_tokens=total_tokens,
            results=results,
        )

    def _compute_scores(
        self, queries: list[str], documents: list[str], batch_size: int, show_progress: bool, **kwargs
    ) -> tuple[np.ndarray, int]:
        """Compute scores for query-document pairs.

        Args:
        ----
            queries: List of query strings.
            documents: List of document strings.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            np.ndarray: Computed scores.
            int: Total token count.

        """
        total_tokens = 0

        if len(queries) < batch_size:
            features = self.pre_process(text=queries, text_pair=documents, **kwargs)
            scores = np.array(self.forward(**features)["scores"])
            total_tokens = features["input_ids"].shape[0] * features["input_ids"].shape[1]
            return scores, total_tokens

        scores = np.zeros(len(queries), dtype=np.float32)
        for docs, org_indices in tqdm(
            bucket_batch_iter(documents, batch_size),
            desc="Ranking",
            disable=not show_progress,
            total=(len(queries) + batch_size - 1) // batch_size,
        ):
            features = self.pre_process(text=[queries[i] for i in org_indices], text_pair=docs, **kwargs)
            scores[org_indices] = np.array(self.forward(**features)["scores"])
            total_tokens += features["input_ids"].shape[0] * features["input_ids"].shape[1]

        return scores, total_tokens
