from __future__ import annotations

from typing import Any

from ofen.models.base.cross_encoder import BaseCrossEncoder
from ofen.models.base.model import BaseModel
from ofen.models.base.text_encoder import BaseTextEncoder


class ModelChecker:
    """A utility class for checking the type of model instances.

    This class provides static methods to determine if a given model
    is a text encoder or a cross encoder.
    """

    @staticmethod
    def is_text_encoder(model: Any) -> bool:
        """Check if the given model is a text encoder.

        Args:
        ----
            model (Any): The model instance to check.

        Returns:
        -------
            bool: True if the model is a BaseTextEncoder instance, False otherwise.

        """
        return isinstance(model, BaseTextEncoder)

    @staticmethod
    def is_cross_encoder(model: Any) -> bool:
        """Check if the given model is a cross encoder.

        Args:
        ----
            model (Any): The model instance to check.

        Returns:
        -------
            bool: True if the model is a BaseCrossEncoder instance, False otherwise.

        """
        return isinstance(model, BaseCrossEncoder)

    @staticmethod
    def is_valid_model(model: Any) -> bool:
        """Check if the given model is a valid Ofen model.

        Args:
        ----
            model (Any): The model instance to check.

        Returns:
        -------
            bool: True if the model is a BaseModel instance, False otherwise.

        """
        return isinstance(model, BaseModel)
