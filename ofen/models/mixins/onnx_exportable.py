from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import torch

from ofen.common.utils import ensure_dir_exists

if TYPE_CHECKING:
    from ofen.types import ModelFeatures, ModelOutputs


class OnnxExportable(abc.ABC):
    @abc.abstractmethod
    def export_to_onnx(self, path: str | None = None) -> str:
        """Abstract method to export the model to ONNX format.

        Args:
        ----
            path (Optional[str]): The path to save the ONNX model. If None, a default path is used.

        Returns:
        -------
            str: The path where the ONNX model was saved.

        """

    def _export_to_onnx(self, inputs: Any, path: str | None = None) -> str:
        """Export the model to ONNX format.

        Args:
        ----
            inputs (Any): The inputs to the model.
            path (Optional[str]): The path to save the ONNX model. If None, a default path is used.

        Returns:
        -------
            str: The path where the ONNX model was saved.

        Raises:
        ------
            AttributeError: If self.config.name_or_path is not defined.
            RuntimeError: If there's an error during the ONNX export process.

        """
        try:
            from ofen.common.onnx_utils import OnnxUtilities

            output_path = path or OnnxUtilities.get_onnx_path(self.config.name_or_path)
            ensure_dir_exists(output_path)

            processed_inputs: ModelFeatures = self.pre_process(inputs)
            outputs: ModelOutputs = self.forward(**processed_inputs)

            dynamic_axes = self._get_dynamic_axes(processed_inputs, outputs)

            with torch.no_grad():
                torch.onnx.export(
                    self,
                    tuple(processed_inputs.values()),
                    output_path,
                    input_names=list(processed_inputs.keys()),
                    output_names=list(outputs.keys()),
                    dynamic_axes=dynamic_axes,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                )

            return output_path
        except AttributeError as ae:
            msg = "self.config.name_or_path is not defined."
            raise AttributeError(msg) from ae
        except Exception as e:
            msg = f"Error during ONNX export: {e!s}"
            raise RuntimeError(msg) from e

    @staticmethod
    def _get_dynamic_axes(inputs: ModelFeatures, outputs: ModelOutputs) -> dict[str, dict[int, str]]:
        """Generate dynamic axes for ONNX export.

        Args:
        ----
            inputs (ModelFeatures): The processed inputs to the model.
            outputs (ModelOutputs): The outputs from the model.

        Returns:
        -------
            Dict[str, Dict[int, str]]: A dictionary of dynamic axes for ONNX export.

        """
        return {
            **{key: {0: "batch", 1: "sequence"} for key in inputs},
            **{key: {0: "batch"} for key in outputs},
        }
