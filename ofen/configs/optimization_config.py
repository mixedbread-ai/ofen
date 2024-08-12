from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ofen.configs.base.base_config import BaseConfig
from ofen.enums import Quantization

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclasses.dataclass
class OptimizationConfig(BaseConfig):
    use_bettertransformer: bool = False
    quantization: Quantization = Quantization.NONE

    int8_ranges: NDArray | None = None
    calibration_data: NDArray | None = None

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> OptimizationConfig:
        return cls._from_registry(model_name) or cls(name_or_path=model_name)
