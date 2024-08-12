from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from ofen.enums import ActivationStrategy

ActivationFunction = Callable[[Tensor], Tensor]

ACTIVATION_FN_BY_STRATEGY: dict[ActivationStrategy, ActivationFunction] = {
    ActivationStrategy.SIGMOID: torch.sigmoid,
    ActivationStrategy.SOFTMAX: lambda x: torch.softmax(x, dim=-1),
    ActivationStrategy.TANH: torch.tanh,
    ActivationStrategy.RELU: torch.relu,
    ActivationStrategy.NONE: lambda x: x,
}


class ActivationLayer(nn.Module):
    """Applies activation function to the input tensor.

    This layer supports various activation strategies defined in the ActivationStrategy enum.
    It can be initialized with either a string representation of the strategy or an
    ActivationStrategy enum value.

    Args:
    ----
        strategy (Union[str, ActivationStrategy]): The activation strategy to apply.

    Attributes:
    ----------
        strategy (ActivationStrategy): The resolved activation strategy.
        fn (ActivationFunction): The activation function corresponding to the strategy.

    Raises:
    ------
        ValueError: If an invalid activation strategy is provided.

    """

    def __init__(self, strategy: str | ActivationStrategy) -> None:
        super().__init__()
        self.strategy = ActivationStrategy(strategy)
        self.fn = ACTIVATION_FN_BY_STRATEGY.get(self.strategy)
        if self.fn is None:
            msg = f"No activation function found for strategy: {self.strategy}"
            raise ValueError(msg)

    def forward(self, logits: Tensor, **kwargs: Any) -> Tensor:
        """Applies the activation function to the input tensor.

        Args:
        ----
            logits (Tensor): The input tensor to be activated.
            **kwargs: Additional keyword arguments (unused, but allowed for flexibility).

        Returns:
        -------
            Tensor: The activated tensor with the last dimension squeezed.

        """
        return self.fn(logits).squeeze(-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.strategy})"
