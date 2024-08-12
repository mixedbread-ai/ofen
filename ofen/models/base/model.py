from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from ofen.pretrainable import Pretrainable

if TYPE_CHECKING:
    from ofen.processors.base import BaseProcessor
    from ofen.types import ModelFeatures, ModelOutputs


class BaseModel(abc.ABC, Pretrainable):
    """Abstract base class for all models in the ofen framework.

    This class provides a common interface for preprocessing, forward pass,
    and postprocessing of model inputs and outputs.

    Attributes
    ----------
        processor (BaseProcessor): The processor used for pre and post processing.
        name_or_path (str): The name or path of the model.

    """

    def __init__(self, name_or_path: str, processor: BaseProcessor):
        """Initialize the BaseModel with a processor.

        Args:
        ----
            name_or_path (str): The name or path of the model.
            processor (BaseProcessor): The processor to use for pre and post processing.

        Raises:
        ------
            ValueError: If processor is None or not an instance of BaseProcessor.

        """
        self.processor = processor
        self.name_or_path = name_or_path

    def pre_process(self, *args: Any, **kwargs: Any) -> ModelFeatures:
        """Preprocess the input data.

        Args:
        ----
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
        -------
            ModelFeatures: The preprocessed features ready for model input.

        """
        return self.processor.pre_process(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, **features: ModelFeatures) -> ModelOutputs:
        """Perform the forward pass of the model.

        Args:
        ----
            **features: The preprocessed features.

        Returns:
        -------
            ModelOutputs: The raw output of the model.

        Raises:
        ------
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError

    def post_process(self, inputs: ModelFeatures, outputs: ModelOutputs) -> ModelOutputs:
        """Postprocess the model outputs.

        Args:
        ----
            inputs (ModelFeatures): The original input features.
            outputs (ModelOutputs): The raw model outputs.

        Returns:
        -------
            ModelOutputs: The postprocessed model outputs.

        """
        return self.processor.post_process(inputs, outputs)

    def pipe(self, *args: Any, **kwargs: Any) -> ModelOutputs:
        """Run the full pipeline: preprocessing, forward pass, and postprocessing.

        Args:
        ----
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
        -------
            ModelOutputs: The final processed outputs of the model.

        """
        features = self.pre_process(*args, **kwargs)
        outputs = self.forward(**features)
        return self.post_process(features, outputs)
