from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ofen.common.utils import auto_device, ensure_import
from ofen.constants import DEFAULT_CACHE_DIR
from ofen.enums import ORTOptimizationLevel, Quantization
from ofen.exceptions import OrtExecutionProviderError
from ofen.logger import LOGGER

with ensure_import("ofen[onnx]"):
    import onnx
    import onnxconverter_common
    import onnxruntime as ort
    import onnxsim
    from onnxruntime import quantization as ort_quantization


class OnnxUtilities:
    """Utility class for ONNX model operations and optimizations."""

    GPU_PROVIDERS = ["CUDAExecutionProvider", "TensorRTExecutionProvider"]
    CPU_PROVIDERS = ["CPUExecutionProvider", "CoreMLExecutionProvider"]

    @staticmethod
    def available_providers(device: str | None = None) -> list[str | tuple[str, dict[str, Any]]]:
        """Get available ONNX Runtime execution providers based on the specified device.

        Args:
        ----
            device (Optional[str]): The target device (CPU, CUDA, or None for auto-detection).

        Returns:
        -------
            List[Union[str, Tuple[str, Dict[str, Any]]]]: List of available providers.

        Raises:
        ------
            OrtExecutionProviderError: If no suitable providers are available.

        """
        available = ort.get_available_providers()

        if device is None:
            preferences = OnnxUtilities.GPU_PROVIDERS + OnnxUtilities.CPU_PROVIDERS
            filtered_preferences = [provider for provider in preferences if provider in available]
            if filtered_preferences:
                return filtered_preferences
            if available:
                return available
            msg = "No execution providers are available."
            raise OrtExecutionProviderError(msg)

        if device.startswith("cuda"):
            if not any(provider in available for provider in OnnxUtilities.GPU_PROVIDERS):
                msg = (
                    f"GPU providers are not available. Consider installing `onnxruntime-gpu` "
                    f"and ensure CUDA is available on your system. Currently installed: {available}"
                )
                raise OrtExecutionProviderError(msg)
            provider_options = {"device_id": int(device.split(":")[-1])} if ":" in device else {}
            cpu_provider = (
                ("CoreMLExecutionProvider", {})
                if "CoreMLExecutionProvider" in available
                else ("CPUExecutionProvider", {})
            )
            return [(provider, provider_options) for provider in OnnxUtilities.GPU_PROVIDERS] + [cpu_provider]

        if device.startswith(("cpu", "mps")):
            if not any(provider in available for provider in OnnxUtilities.CPU_PROVIDERS):
                msg = (
                    f"CPU providers are not available. Consider installing `onnxruntime` "
                    f"and ensure OpenVINO and CoreML are available on your system. Currently installed: {available}"
                )
                raise OrtExecutionProviderError(msg)
            return available

        msg = f"Device {device} is not supported"
        raise OrtExecutionProviderError(msg)

    @staticmethod
    def get_onnx_session(
        onnx_model_path: str,
        *,
        device: str = auto_device(),
        ort_optimization_level: ORTOptimizationLevel = ORTOptimizationLevel.O2,
        enable_mem_reuse: bool = True,
        enable_mem_pattern: bool = True,
    ) -> ort.InferenceSession:
        """Create an ONNX Runtime InferenceSession for the given model path and device.

        Args:
        ----
            onnx_model_path (str): Path to the ONNX model file.
            device (str): Target device for inference.
            ort_optimization_level (ORTOptimizationLevel): Optimization level for the session.
            enable_mem_reuse (bool): Whether to enable memory reuse.
            enable_mem_pattern (bool): Whether to enable memory pattern.

        Returns:
        -------
            ort.InferenceSession: Configured ONNX Runtime session.

        Raises:
        ------
            FileNotFoundError: If the ONNX model file is not found.
            OrtExecutionProviderError: If no suitable providers are available.

        """
        if not os.path.exists(onnx_model_path):
            msg = f"ONNX model file not found: {onnx_model_path}"
            raise FileNotFoundError(msg)

        try:
            providers = OnnxUtilities.available_providers(device)
        except OrtExecutionProviderError as e:
            LOGGER.error(f"Error getting available providers: {e!s}")
            raise

        session_options = ort.SessionOptions()
        session_options.enable_mem_reuse = enable_mem_reuse
        session_options.enable_mem_pattern = enable_mem_pattern
        session_options.graph_optimization_level = ort_optimization_level.to_ort()

        try:
            return ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=providers)
        except Exception as e:
            LOGGER.error(f"Error creating ONNX Runtime session: {e!s}")
            raise

    @staticmethod
    def get_onnx_path(
        name_or_path: str,
        quantization: Quantization | None = None,
        simplified: bool = False,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> str:
        """Generate the path for an ONNX model file based on the given parameters.

        Args:
        ----
            name_or_path (str): Name or path of the model.
            quantization (Optional[Quantization]): Quantization type.
            simplified (bool): Whether the model is simplified.
            cache_dir (str): Cache directory for ONNX models.

        Returns:
        -------
            str: Full path to the ONNX model file.

        """
        name_as_path = Path(name_or_path)
        file_name = "model" if not name_as_path.name.endswith(".onnx") else name_as_path.name.replace(".onnx", "")
        if quantization:
            file_name += f"_{quantization.value}"
        if simplified:
            file_name += "_simplified"
        file_name += ".onnx"

        return os.path.join(
            cache_dir,
            name_as_path.parent.name,
            file_name,
        )

    @staticmethod
    def convert_to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
        """Convert ONNX model to FP16 precision.

        Args:
        ----
            model (onnx.ModelProto): Input ONNX model.

        Returns:
        -------
            onnx.ModelProto: FP16 converted ONNX model.

        """
        return onnxconverter_common.float16.convert_float_to_float16(model)

    @staticmethod
    def convert_to_int8(model: onnx.ModelProto) -> onnx.ModelProto:
        """Convert ONNX model to INT8 precision.

        Args:
        ----
            model (onnx.ModelProto): Input ONNX model.

        Returns:
        -------
            onnx.ModelProto: INT8 converted ONNX model.

        """
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            try:
                onnx.save(model, temp.name)
                ort_quantization.quantize_dynamic(temp.name, temp.name)
                return onnx.load(temp.name)
            finally:
                os.unlink(temp.name)

    @staticmethod
    def quantize_model(model: onnx.ModelProto, quantization: Quantization) -> onnx.ModelProto:
        """Quantize the ONNX model based on the specified quantization type.

        Args:
        ----
            model (onnx.ModelProto): Input ONNX model.
            quantization (Quantization): Desired quantization type.

        Returns:
        -------
            onnx.ModelProto: Quantized ONNX model.

        """
        model = ort_quantization.quant_pre_process(model)
        if quantization == Quantization.INT8:
            return OnnxUtilities.convert_to_int8(model)
        if quantization == Quantization.FP16:
            return OnnxUtilities.convert_to_fp16(model)
        return model

    @staticmethod
    def simplify_model(model: onnx.ModelProto) -> onnx.ModelProto:
        """Simplify the ONNX model using onnxsim if available.

        Args:
        ----
            model (onnx.ModelProto): Input ONNX model.

        Returns:
        -------
            onnx.ModelProto: Simplified ONNX model.

        """
        new_model, check = onnxsim.simplify(model)
        if not check:
            LOGGER.warning(
                "Model was optimized, but not within the specified constraints. "
                "Continuing without the optimized model."
            )
            return model
        return new_model

    @staticmethod
    def optimize(name_or_path: str, quantization: Quantization = Quantization.NONE, simplify: bool = False) -> str:
        """Optimize the ONNX model by quantizing and/or simplifying it.

        Args:
        ----
            name_or_path (str): Name or path of the input ONNX model.
            quantization (Quantization): Desired quantization type.
            simplify (bool): Whether to simplify the model.

        Returns:
        -------
            str: Path to the optimized ONNX model.

        Raises:
        ------
            FileNotFoundError: If the input ONNX file is not found.

        """
        output_path = OnnxUtilities.get_onnx_path(name_or_path, quantization=quantization, simplified=simplify)
        if Path(output_path).exists():
            return output_path

        input_path = OnnxUtilities.get_onnx_path(name_or_path)
        if not Path(input_path).exists():
            msg = f"ONNX file not found at {input_path}."
            raise FileNotFoundError(msg)

        try:
            model = onnx.load(input_path)
            if simplify:
                model = OnnxUtilities.simplify_model(model)
            if quantization:
                model = OnnxUtilities.quantize_model(model, quantization)

            onnx.save(model, output_path)
            return output_path
        except Exception as e:
            LOGGER.error(f"Error optimizing ONNX model: {e!s}")
            raise
