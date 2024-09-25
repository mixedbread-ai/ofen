from __future__ import annotations

from ofen.models.base.cross_encoder import BaseCrossEncoder, CrossEncoderOutput
from ofen.models.base.ort_model import BaseORTModel
from ofen.processors import TextProcessor


class ORTCrossEncoder(BaseORTModel, BaseCrossEncoder):
    """ONNX Runtime Cross Encoder model.

    This class implements a cross encoder using ONNX Runtime for inference.
    It inherits from BaseORTModel and BaseCrossEncoder.
    """

    def __init__(
        self,
        name_or_path: str,
        processor_kwargs: dict | None = None,
        processor: TextProcessor | None = None,
        **kwargs,
    ) -> None:
        """Initialize the ORTCrossEncoder.

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
            processor = TextProcessor.from_pretrained(
                name_or_path, return_tensors="np", return_token_type_ids=True, **processor_kwargs
            )

        super().__init__(name_or_path, processor=processor, **kwargs)

    def forward(self, **kwargs) -> CrossEncoderOutput:
        """Perform forward pass through the model.

        Args:
        ----
            **kwargs: Input tensors as keyword arguments.

        Returns:
        -------
            CrossEncoderOutput: The output of the cross encoder.

        """
        out = super().forward(**kwargs)
        return CrossEncoderOutput(scores=out[0])

    @property
    def _auto_loading_cls(self) -> type[BaseCrossEncoder]:
        """Get the auto-loading class for this model.

        Returns
        -------
            Type['CrossEncoder']: The CrossEncoder class for auto-loading.

        """
        from ofen.models.torch.cross_encoder import CrossEncoder

        return CrossEncoder
