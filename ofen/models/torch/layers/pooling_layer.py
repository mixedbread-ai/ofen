from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from ofen.enums import PoolingStrategy


def pool_cls(token_embeddings: Tensor, _: Tensor) -> Tensor:
    """Pools the [CLS] token.

    Args:
    ----
        token_embeddings (Tensor): The token embeddings.
        _ (Tensor): Unused attention mask.

    Returns:
    -------
        Tensor: The [CLS] token embedding.

    """
    return token_embeddings[:, 0]


def pool_max(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the maximum value from the token embeddings, ignoring padding tokens.

    Args:
    ----
        token_embeddings (Tensor): The token embeddings.
        attention_mask (Tensor): The attention mask.

    Returns:
    -------
        Tensor: The maximum pooled embeddings.

    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to a large negative value
    return torch.max(token_embeddings, 1)[0]


def pool_mean_sqrt_len(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the mean of the token embeddings, scaled by the square root of the sequence length.

    Args:
    ----
        token_embeddings (Tensor): The token embeddings.
        attention_mask (Tensor): The attention mask.

    Returns:
    -------
        Tensor: The mean pooled embeddings, scaled by sqrt(sequence length).

    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(attention_mask.sum(1, keepdim=True), min=1e-9)
    return sum_embeddings / torch.sqrt(sum_mask)


def pool_weighted_mean(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the weighted mean of the token embeddings, with weights increasing linearly.

    Args:
    ----
        token_embeddings (Tensor): The token embeddings.
        attention_mask (Tensor): The attention mask.

    Returns:
    -------
        Tensor: The weighted mean pooled embeddings.

    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    seq_len = token_embeddings.shape[1]
    weights = torch.arange(start=1, end=seq_len + 1, dtype=token_embeddings.dtype, device=token_embeddings.device)
    weights = weights.unsqueeze(0).unsqueeze(-1).expand(token_embeddings.size())
    input_mask_expanded = input_mask_expanded * weights
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def pool_last(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the last valid token embedding in each sequence.

    Args:
    ----
        token_embeddings (Tensor): The token embeddings.
        attention_mask (Tensor): The attention mask.

    Returns:
    -------
        Tensor: The last valid token embeddings.

    """
    indices = torch.sum(attention_mask, dim=1, keepdim=True) - 1
    indices = indices.unsqueeze(-1).expand(-1, -1, token_embeddings.shape[-1])
    return token_embeddings.gather(1, indices).squeeze(1)


def pool_sum(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the sum of the token embeddings, ignoring padding tokens.

    Args:
    ----
        token_embeddings (Tensor): The token embeddings.
        attention_mask (Tensor): The attention mask.

    Returns:
    -------
        Tensor: The sum pooled embeddings.

    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    return torch.sum(token_embeddings * input_mask_expanded, 1)


def pool_mean(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the mean of the token embeddings, ignoring padding tokens.

    Args:
    ----
        token_embeddings (Tensor): The token embeddings.
        attention_mask (Tensor): The attention mask.

    Returns:
    -------
        Tensor: The mean pooled embeddings.

    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


POOLING_FUNCTIONS: dict[PoolingStrategy, Callable[[Tensor, Tensor], Tensor]] = {
    PoolingStrategy.CLS: pool_cls,
    PoolingStrategy.MAX: pool_max,
    PoolingStrategy.MEAN: pool_mean,
    PoolingStrategy.MEAN_SQRT_LEN: pool_mean_sqrt_len,
    PoolingStrategy.WEIGHTED_MEAN: pool_weighted_mean,
    PoolingStrategy.LAST: pool_last,
    PoolingStrategy.SUM: pool_sum,
    PoolingStrategy.NONE: lambda token_embeddings, _: token_embeddings,
}


class PoolingLayer(nn.Module):
    """Performs pooling on the token embeddings.

    This layer generates a fixed-size sentence embedding from variable-sized token embeddings.
    It supports various pooling strategies, including using the CLS token if available.

    Attributes:
    ----------
        strategy (PoolingStrategy): The pooling strategy to apply.
        fn (Callable): The pooling function corresponding to the strategy.

    Args:
    ----
        strategy (Union[str, PoolingStrategy]): The pooling strategy to apply.

    Raises:
    ------
        ValueError: If an invalid pooling strategy is provided.

    """

    def __init__(self, strategy: str | PoolingStrategy) -> None:
        super().__init__()
        self.strategy = PoolingStrategy(strategy)
        self.fn = POOLING_FUNCTIONS.get(self.strategy)
        if self.fn is None:
            msg = f"Invalid pooling strategy: {strategy}"
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pooling_strategy={self.strategy})"

    def set_pooling_strategy(self, strategy: str | PoolingStrategy) -> None:
        """Sets a new pooling strategy for the layer.

        Args:
        ----
            strategy (Union[str, PoolingStrategy]): The new pooling strategy to apply.

        Raises:
        ------
            ValueError: If an invalid pooling strategy is provided.

        """
        strategy = PoolingStrategy(strategy)
        fn = POOLING_FUNCTIONS.get(strategy)
        if fn is None:
            msg = f"Invalid pooling strategy: {strategy}"
            raise ValueError(msg)
        self.fn = fn
        self.strategy = strategy

    def forward(self, attention_mask: Tensor, last_hidden_state: Tensor, **kwargs: Any) -> Tensor:
        """Applies the pooling function to the token embeddings.

        Args:
        ----
            attention_mask (Tensor): Attention mask to ignore padding tokens.
            last_hidden_state (Tensor): The last hidden states of the token embeddings.
            **kwargs: Additional keyword arguments (unused, but allowed for flexibility).

        Returns:
        -------
            Tensor: Pooled sentence embedding.

        """
        return self.fn(last_hidden_state, attention_mask)
