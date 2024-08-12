from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ofen.types import ModelFeatures, ModelOutputs


class BaseProcessor(abc.ABC):
    @abc.abstractmethod
    def pre_process(self, *args, **kwargs: Any) -> ModelFeatures:
        """Pre-process input data before feeding it to the model.

        This method prepares a single input or a small batch of inputs for the model.
        It transforms raw input data into the format expected by the model.

        Args:
        ----
            *args: Variable length argument list containing the input data.
            **kwargs: Additional keyword arguments for pre-processing.

        Returns:
        -------
            ModelFeatures: The pre-processed input data, ready to be fed into the model.

        Example:
        -------
            features = processor.pre_process(text)
            model_output = model(features)

        """

    @abc.abstractmethod
    def post_process(self, inputs: ModelFeatures, outputs: ModelOutputs) -> ModelOutputs:
        """Post-process model outputs before returning them.

        This method takes the raw outputs from the model and the corresponding inputs,
        and performs any necessary post-processing to prepare the final results.

        Args:
        ----
            inputs (ModelFeatures): The input data used for inference.
            outputs (ModelOutputs): The raw outputs from the model.

        Returns:
        -------
            ModelOutputs: The post-processed model outputs, ready for final use.

        Example:
        -------
            features = processor.pre_process(text)
            raw_outputs = model(features)
            final_outputs = processor.post_process(features, raw_outputs)

        """
