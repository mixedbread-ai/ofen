from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Callable, TypeVar

from ofen.batch_processor.generator.batch_processor import BatchProcessor, BatchProcessorConfig
from ofen.logger import LOGGER
from ofen.runners.base import RunnerRequestInfo

if TYPE_CHECKING:
    from ofen.models.base.model import BaseModel

T = TypeVar("T", bound="BaseModel")


def make_engine(batch_config: BatchProcessorConfig | None = None) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Defines an engine class from a model class.

    Args:
    ----
        batch_config (Optional[BatchProcessorConfig]): Configuration for batch processing.
            Defaults to None.

    Returns:
    -------
        Callable[[Type[BaseModel]], Type[BaseModel]]: A decorator function that creates an engine class.

    Example:
    -------
    ```python
    @make_engine()
    class MyModel(BaseModel):
        pass
    ```

    """
    batch_config = batch_config or BatchProcessorConfig()

    def decorator(model_cls: type[BaseModel]) -> type[BaseModel]:
        if inspect.isclass(model_cls):
            return Runner.from_model_cls(
                model_cls,
                batch_config=batch_config,
            )
        msg = f"make_engine expects a class, got {type(model_cls)}"
        raise TypeError(msg)

    return decorator


class Runner:
    """A class for creating and managing model runners with optional metrics collection."""

    REQUIRED_METHODS = ("forward",)

    @staticmethod
    def with_metrics(func: Callable) -> Callable:
        """Decorator to mark methods for metric collection.
        This decorator is only used in an engine context and has no effect outside of it.

        Args:
        ----
            func (Callable): The function to be decorated.

        Returns:
        -------
            Callable: The decorated function.

        """
        setattr(func, "__with_metrics", True)
        return func

    @classmethod
    def from_model_cls(
        cls,
        model_cls: type[T],
        batch_config: BatchProcessorConfig | None = BatchProcessorConfig(),
    ) -> type[T]:
        """Create a derived class from a model class with optional batch processing.

        Args:
        ----
            model_cls (Type[T]): The base model class.
            batch_config (Optional[BatchProcessorConfig]): Configuration for batch processing.
                Defaults to None.

        Returns:
        -------
            Type[T]: A derived class with optional batch processing capabilities.

        """
        cls._validate_model_class(model_cls)
        DerivedClass = cls._create_derived_class(model_cls, batch_config)
        cls._prepare_decorators(model_cls, DerivedClass)
        return DerivedClass

    @classmethod
    def _create_derived_class(cls, model_cls: type[T], batch_config: BatchProcessorConfig | None = None) -> type[T]:
        """Create a derived class from a model class with optional batch processing.

        Args:
        ----
            model_cls (Type[T]): The base model class.
            batch_config (Optional[BatchProcessorConfig]): Configuration for batch processing.
                Defaults to None.

        Returns:
        -------
            Type[T]: A derived class with optional batch processing capabilities.

        """

        class DerivedClass(model_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                if batch_config:
                    self.forward = BatchProcessor.from_config(batch_config, super().forward)
                    LOGGER.info(
                        f"Initialized {self.__class__.__name__} with dynamic batching and config: {batch_config}"
                    )
                else:
                    LOGGER.info(f"Initialized {self.__class__.__name__} without dynamic batching")

        DerivedClass.__module__ = model_cls.__module__
        DerivedClass.__name__ = f"{model_cls.__name__}Engine"
        DerivedClass.__qualname__ = f"{model_cls.__name__}Engine"
        return DerivedClass

    @classmethod
    def _prepare_decorators(cls, model_cls: type[T], DerivedClass: type[T]) -> None:
        """Prepare decorators for the derived class.

        Args:
        ----
            model_cls (Type[T]): The base model class.
            DerivedClass (Type[T]): The derived class.

        """
        for attr_name in dir(model_cls):
            func = getattr(model_cls, attr_name)
            if callable(func) and getattr(func, "__with_metrics", False):
                setattr(DerivedClass, attr_name, cls._wrap_with_metrics(func))

    @classmethod
    def _wrap_with_metrics(cls, func: Callable) -> Callable:
        """Wrap a function with metrics collection.

        Args:
        ----
            func (Callable): The function to be wrapped.

        Returns:
        -------
            Callable: The wrapped function with metrics collection.

        """
        func_params = inspect.signature(func).parameters
        has_usage = "return_usage" in func_params

        @functools.wraps(func)
        def wrapped_func(self, *args, return_metrics: bool = True, **kwargs):
            if not return_metrics or not isinstance(self.forward, BatchProcessor):
                return func(self, *args, **kwargs)

            with RunnerRequestInfo() as req:
                req.batch_stats_in = self.forward.stats()
                if has_usage:
                    result = func(self, *args, return_usage=True, **kwargs)
                    req.usage = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                else:
                    result = func(self, *args, **kwargs)
                    req.usage = None
                req.batch_stats_out = self.forward.stats()

            return result, req

        return wrapped_func

    @classmethod
    def _validate_model_class(cls, model_cls: type[BaseModel]) -> None:
        """Validate that the model class has all required methods.

        Args:
        ----
            model_cls (Type[BaseModel]): The model class to validate.

        Raises:
        ------
            TypeError: If the model class is missing required methods.

        """
        missing_methods = [
            method
            for method in cls.REQUIRED_METHODS
            if not hasattr(model_cls, method) or not callable(getattr(model_cls, method))
        ]
        if missing_methods:
            method_type = "class methods" if all(m.startswith("from_") for m in missing_methods) else "methods"
            msg = (
                f"Model class {model_cls.__name__} must have the following {method_type}: "
                f"{', '.join(missing_methods)}"
            )
            raise TypeError(msg)
