from __future__ import annotations

import numpy as np
import simsimd


def top_k_numpy(scores: np.ndarray, k: int, sort: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Get the top k scores and their indices from a numpy array.

    Args:
    ----
        scores (np.ndarray): Input array of scores.
        k (int): Number of top elements to return.
        sort (bool, optional): Whether to sort the results by score. Otherwise they will be sorted by index.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: Top k scores and their indices.

    Raises:
    ------
        ValueError: If input is not a numpy array or k is not a positive integer.

    """
    assert isinstance(k, int) and k > 0, "Input 'k' must be a positive integer."

    k = min(k, len(scores))

    # Get the indices of the top k elements
    top_k_indices = np.argpartition(scores, -k)[-k:]

    if sort:
        # Extract the top k elements using the indices
        top_k_scores = scores[top_k_indices]
        # Sort the top k elements and their indices
        sorted_indices = np.argsort(top_k_scores)[::-1]
        top_k_scores = top_k_scores[sorted_indices]
        top_k_indices = top_k_indices[sorted_indices]
    else:
        # sort the indices first
        top_k_indices = np.sort(top_k_indices)
        top_k_scores = scores[top_k_indices]

    return top_k_scores, top_k_indices


def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the inner product between two sets of embeddings.
    Supports one-to-one and one-to-many comparisons.

    Args:
    ----
        a (np.ndarray): First set of embeddings. Shape: (n, d) or (1, d)
        b (np.ndarray): Second set of embeddings. Shape: (m, d) or (1, d)

    Returns:
    -------
        np.ndarray: Inner product results. Shape: (n, m) or (1, 1)

    """
    return np.array(simsimd.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity between two sets of embeddings.
    Supports one-to-one and one-to-many comparisons.

    Args:
    ----
        a (np.ndarray): First set of embeddings. Shape: (n, d) or (1, d)
        b (np.ndarray): Second set of embeddings. Shape: (m, d) or (1, d)

    Returns:
    -------
        np.ndarray: Cosine similarity results. Shape: (n, m) or (1, 1)

    """
    return np.array(simsimd.cosine(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance between two sets of embeddings.
    Supports one-to-one and one-to-many comparisons.

    Args:
    ----
        a (np.ndarray): First set of embeddings. Shape: (n, d) or (1, d)
        b (np.ndarray): Second set of embeddings. Shape: (m, d) or (1, d)

    Returns:
    -------
        np.ndarray: Euclidean distance results. Shape: (n, m) or (1, 1)

    """
    return np.array(simsimd.sqeuclidean(a, b))


def jensen_shannon_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Jensen-Shannon distance between two sets of embeddings.
    Supports one-to-one and one-to-many comparisons.

    Args:
    ----
        a (np.ndarray): First set of embeddings. Shape: (n, d) or (1, d)
        b (np.ndarray): Second set of embeddings. Shape: (m, d) or (1, d)

    Returns:
    -------
        np.ndarray: Jensen-Shannon distance results. Shape: (n, m) or (1, 1)

    """
    return np.array(simsimd.jaccard(a, b))


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Hamming distance between two sets of embeddings.
    Supports one-to-one and one-to-many comparisons.

    Args:
    ----
        a (np.ndarray): First set of embeddings. Shape: (n, d) or (1, d)
        b (np.ndarray): Second set of embeddings. Shape: (m, d) or (1, d)

    Returns:
    -------
        np.ndarray: Hamming distance results. Shape: (n, m) or (1, 1)

    """
    return np.array(simsimd.hamming(a, b))
