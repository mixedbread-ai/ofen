from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ofen.processors.text_processor import TextProcessor

if TYPE_CHECKING:
    from ofen.types import ModelFeatures


class ColbertProcessor(TextProcessor):
    """A processor for ColBERT-style text processing.

    This class extends TextProcessor to handle specific requirements of ColBERT,
    including query and document marker tokens and specialized processing for
    queries and documents.
    """

    def __init__(self, name_or_path: str, *, query_max_length: int = 32, **kwargs: Any) -> None:
        """Initialize the ColbertProcessor.

        Args:
        ----
            name_or_path (str): The name or path of the pre-trained model to use.
            query_max_length (int, optional): Maximum length for query processing. Defaults to 32.
            **kwargs: Additional keyword arguments to pass to the parent TextProcessor.

        Raises:
        ------
            ValueError: If the required special tokens are not found in the tokenizer.

        """
        super().__init__(name_or_path, **kwargs)
        self.query_marker_token_id = self._tokenizer.convert_tokens_to_ids("[Q]")
        self.doc_marker_token_id = self._tokenizer.convert_tokens_to_ids("[D]")
        self.query_max_length = query_max_length

    def _process_queries(self, queries: list[str], full_length_search: bool = False, **kwargs: Any) -> ModelFeatures:
        """Process a list of queries for ColBERT.

        Args:
        ----
            queries (List[str]): List of query strings to process.
            full_length_search (bool, optional): If True, use full length for search. Defaults to False.
            **kwargs: Additional keyword arguments for processing.

        Returns:
        -------
            ModelFeatures: Processed features for the queries.

        """
        feat = super().__call__(
            queries,
            max_length=None if full_length_search else self.query_max_length,
            pad_token_id=self.mask_token_id,
            **kwargs,
        )
        feat["input_ids"][:, 1] = self.query_marker_token_id
        return feat

    def _process_documents(self, documents: list[str], **kwargs: Any) -> ModelFeatures:
        """Process a list of documents for ColBERT.

        Args:
        ----
            documents (List[str]): List of document strings to process.
            **kwargs: Additional keyword arguments for processing.

        Returns:
        -------
            ModelFeatures: Processed features for the documents.

        """
        feat = super().__call__(documents, **kwargs)
        feat["input_ids"][:, 1] = self.doc_marker_token_id
        return feat

    def __call__(self, text: str | list[str], is_query: bool = False, **kwargs: Any) -> ModelFeatures:
        """Process the input text for ColBERT.

        Args:
        ----
            text (Union[str, List[str]]): Input text or list of texts to process.
            is_query (bool, optional): If True, process as queries. Otherwise, process as documents. Defaults to False.
            **kwargs: Additional keyword arguments for processing.

        Returns:
        -------
            ModelFeatures: Processed features for the input text.

        """
        if isinstance(text, str):
            text = [text]

        # Prepend each text with a period and space for consistency
        text = [f". {t}" for t in text]

        if is_query:
            return self._process_queries(text, **kwargs)
        return self._process_documents(text, **kwargs)
