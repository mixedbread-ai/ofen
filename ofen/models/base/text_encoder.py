from __future__ import annotations

from typing import TYPE_CHECKING

from ofen.models.base.encoder import BaseEncoder, EncodingResult, StreamedEncodingItem
from ofen.runners.runner import Runner

if TYPE_CHECKING:
    from collections.abc import Generator

    from ofen.enums import EncodingFormat


class BaseTextEncoder(BaseEncoder):
    @Runner.with_metrics
    def encode(
        self,
        text: str | list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
        **kwargs,
    ) -> EncodingResult:
        """Encode text into embeddings.

        Args:
        ----
            text: Input text or list of texts.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep in the output embeddings.
            encoding_format: Format(s) for quantization.
            framework: Framework to return the embeddings in ("numpy" or "pytorch").
            **kwargs: Additional keyword arguments for pre-processing.

        Returns:
        -------
            TextEncodingResult containing embeddings and total tokens.

        Raises:
        ------
            ValueError: If input text is empty.

        """
        if isinstance(text, str):
            text = [text]

        return self._encode(
            data=text,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=normalize,
            dimensions=dimensions,
            encoding_format=encoding_format,
            **kwargs,
        )

    def stream_encode(
        self,
        text: str | list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        dimensions: int | None = None,
        encoding_format: str | EncodingFormat | list[EncodingFormat] | list[str] | None = None,
        **kwargs,
    ) -> Generator[StreamedEncodingItem, None, None]:
        """Stream embeddings for text.

        Args:
        ----
            text: Input text or list of texts.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            normalize: Whether to normalize the embeddings.
            dimensions: Number of dimensions to keep in the output embeddings.
            encoding_format: Format(s) for quantization.
            framework: Framework to return the embeddings in ("numpy" or "pytorch").
            **kwargs: Additional keyword arguments for pre-processing.

        Yields:
        ------
            StreamedTextEncodingResult or MultiFormatStreamedTextEncodingResult containing generated embeddings and indices for each batch.

        Raises:
        ------
            ValueError: If input text is empty.

        """
        if isinstance(text, str):
            text = [text]

        yield from self._stream_encode(
            data=text,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=normalize,
            dimensions=dimensions,
            encoding_format=encoding_format,
            **kwargs,
        )
