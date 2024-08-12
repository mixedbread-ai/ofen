from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ofen.enums import Quantization
from ofen.logger import LOGGER
from ofen.models.model_helper import to_bettertransformer

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class TorchOptimizer:
    @staticmethod
    def optimize(
        pretrained_model: PreTrainedModel,
        device: str,
        quantization: Quantization | None = None,
        use_bettertransformer: bool = False,
    ) -> PreTrainedModel:
        """Optimize the pretrained model based on the specified configuration.

        Args:
        ----
            pretrained_model (PreTrainedModel): The pretrained model to optimize.
            device (str): The device to use for optimization.
            quantization (Optional[Quantization]): The quantization method to apply, if any.
            use_bettertransformer (bool): Whether to use BetterTransformer optimization.

        Returns:
        -------
            PreTrainedModel: The optimized model.

        """
        if use_bettertransformer:
            pretrained_model = to_bettertransformer(pretrained_model, device)

        if quantization:
            pretrained_model = TorchOptimizer.quantize(pretrained_model, device, quantization)

        return pretrained_model

    @staticmethod
    def quantize(model: PreTrainedModel, device: str, quantization: Quantization) -> PreTrainedModel:
        """Apply quantization to the model based on the specified quantization type and device.

        Args:
        ----
            model (PreTrainedModel): The model to quantize.
            device (str): The device to use for quantization.
            quantization (Quantization): The quantization method to apply.

        Returns:
        -------
            PreTrainedModel: The quantized model.

        Raises:
        ------
            ValueError: If an unsupported quantization method is specified.

        """
        if device.startswith("mps") and quantization == Quantization.INT8:
            LOGGER.warning("Quantization to INT8 is not supported for MPS devices. Skipping quantization.")
            return model

        if device.startswith("cpu") and quantization == Quantization.FP16:
            LOGGER.warning("Quantization to FP16 is not supported for CPU devices. Skipping quantization.")
            return model

        if quantization == Quantization.FP16:
            return model.half()
        elif quantization == Quantization.INT8:
            return TorchOptimizer._apply_int8_quantization(model)

        msg = f"Unsupported quantization method: {quantization}"
        raise ValueError(msg)

    @staticmethod
    def _apply_int8_quantization(model: PreTrainedModel) -> PreTrainedModel:
        """Apply INT8 quantization to the model.

        Args:
        ----
            model (PreTrainedModel): The model to quantize.

        Returns:
        -------
            PreTrainedModel: The quantized model.

        """
        if torch.backends.quantized.engine == "none":
            torch.backends.quantized.engine = "qnnpack"

        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
