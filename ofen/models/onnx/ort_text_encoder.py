from __future__ import annotations

from ofen.models.base.encoder import BaseTextEncoder, EncoderOutput
from ofen.models.base.ort_model import BaseORTModel
from ofen.processors import TextProcessor


class ORTTextEncoder(BaseORTModel, BaseTextEncoder):
    """ONNX Runtime Text Encoder model.

    This class implements a text encoder using ONNX Runtime for inference.
    It inherits from BaseORTModel and BaseTextEncoder.
    """

    def __init__(
        self,
        name_or_path: str,
        processor_kwargs: dict | None = None,
        processor: TextProcessor | None = None,
        **kwargs,
    ) -> None:
        """Initialize the ORTTextEncoder.

        Args:
            name_or_path (str): The name or path of the pre-trained model.
            processor_kwargs (Optional[Dict]): Additional keyword arguments for the processor.
            processor (Optional[TextProcessor]): A pre-configured TextProcessor instance.
            **kwargs: Additional keyword arguments to pass to the base class.

        Raises:
            ValueError: If both processor and processor_kwargs are provided.

        """
        if not processor:
            processor_kwargs = processor_kwargs or {}
            processor = TextProcessor.from_pretrained(name_or_path, return_tensors="np", **processor_kwargs)

        super().__init__(name_or_path=name_or_path, processor=processor, **kwargs)

    def forward(self, **kwargs) -> EncoderOutput:
        """Perform forward pass through the model.

        Args:
            **kwargs: Input tensors as keyword arguments.

        Returns:
            EncoderOutput: The output of the text encoder.

        """
        out = super().forward(**kwargs)
        return EncoderOutput(embeddings=out[0])

    @property
    def _auto_loading_cls(self) -> type[BaseTextEncoder]:
        """Get the auto-loading class for this model.

        Returns
            Type[BaseTextEncoder]: The TextEncoder class for auto-loading.

        """
        from ofen.models.torch.text_encoder import TextEncoder

        return TextEncoder
