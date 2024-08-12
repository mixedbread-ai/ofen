from __future__ import annotations

import abc
from typing import Any, TypeVar

T = TypeVar("T", bound="Pretrainable")


class Pretrainable:
    """Abstract base class for objects that can be initialized from pre-trained models or configurations.

    This class defines the interface for creating instances from pre-trained models or
    configuration objects, which is commonly used in machine learning frameworks.
    """

    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls: type[T], name: str, **kwargs: dict[str, Any]) -> T:
        """Create an instance from a pre-trained model.

        Args:
        ----
            name (str): The name or path of the pre-trained model.
            **kwargs: Additional keyword arguments for model initialization.

        Returns:
        -------
            T: An instance of the class.

        Raises:
        ------
            NotImplementedError: If the method is not implemented by the subclass.

        """
        msg = "Subclasses must implement from_pretrained method"
        raise NotImplementedError(msg)

    @classmethod
    @abc.abstractmethod
    def from_config(cls: type[T], config: Any) -> T:
        """Create an instance from a configuration object.

        Args:
        ----
            config (Any): The configuration object.

        Returns:
        -------
            T: An instance of the class.

        Raises:
        ------
            NotImplementedError: If the method is not implemented by the subclass.

        """
        msg = "Subclasses must implement from_config method"
        raise NotImplementedError(msg)
