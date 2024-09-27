from __future__ import annotations

from typing import Any, Callable

from torch import Tensor, nn

from ofen.common.utils import identity
from ofen.enums import NormalizationStrategy


def l2_normalize(x: Tensor) -> Tensor:
    return nn.functional.normalize(x, p=2, dim=-1)


def l1_normalize(x: Tensor) -> Tensor:
    return nn.functional.normalize(x, p=1, dim=-1)


# Define normalization functions for each strategy
NORMALIZATION_FN_BY_STRATEGY: dict[NormalizationStrategy, Callable[[Tensor], Tensor]] = {
    NormalizationStrategy.L2: l2_normalize,
    NormalizationStrategy.L1: l1_normalize,
    NormalizationStrategy.NONE: identity,
}


class NormalizationLayer(nn.Module):
    """A PyTorch module that performs normalization on input tensors.

    This layer supports multiple normalization strategies, including L1, L2,
    and no normalization. The strategy can be specified during initialization.

    Attributes:
    ----------
        strategy (NormalizationStrategy): The selected normalization strategy.
        norm_fn (Callable[[Tensor], Tensor]): The normalization function corresponding
            to the selected strategy.

    Args:
    ----
        strategy (Union[str, NormalizationStrategy], optional): The normalization strategy to apply.
            Defaults to NormalizationStrategy.L2.

    Raises:
    ------
        ValueError: If an unsupported normalization strategy is provided.

    Example:
    -------
        >>> layer = NormalizationLayer(strategy="L2")
        >>> input_tensor = torch.randn(10, 5)
        >>> normalized_tensor = layer(input_tensor)

    """

    def __init__(self, strategy: str | NormalizationStrategy = NormalizationStrategy.L2) -> None:
        super().__init__()
        self.strategy = NormalizationStrategy(strategy)
        self.norm_fn = NORMALIZATION_FN_BY_STRATEGY.get(self.strategy)
        if self.norm_fn is None:
            msg = f"Unsupported normalization strategy: {strategy}"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.strategy})"

    def forward(self, input_tensor: Tensor, **kwargs: Any) -> Tensor:
        """Applies the normalization function to the input tensor.

        Args:
        ----
            input_tensor (Tensor): The tensor to normalize.
            **kwargs: Additional keyword arguments (unused, but allowed for flexibility).

        Returns:
        -------
            Tensor: The normalized tensor.

        Raises:
        ------
            ValueError: If the input tensor is not of the expected shape or type.

        """
        return self.norm_fn(input_tensor)
