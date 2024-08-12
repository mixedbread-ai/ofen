from __future__ import annotations

import dataclasses
import json
import os

from huggingface_hub import hf_hub_download, list_repo_files

from ofen.common.utils import first
from ofen.constants import DEFAULT_CACHE_DIR
from ofen.enums import PoolingStrategy
from ofen.logger import LOGGER


@dataclasses.dataclass
class SbertConfig:
    """Configuration for SentenceBERT models."""

    pooling_strategy: PoolingStrategy
    embedding_dim: int
    max_seq_length: int
    normalize: bool


def is_sentence_transformer(model_name_or_path: str, **kwargs) -> bool:
    """Check if the given model is a SentenceTransformer model.

    Args:
    ----
        model_name_or_path (str): Path or name of the model.
        **kwargs: Additional arguments for downloading.

    Returns:
    -------
        bool: True if it is a SentenceTransformer model, False otherwise.

    """
    try:
        return bool(download_file_from_hf(model_name_or_path, "modules.json", **kwargs))
    except Exception:
        return False


def get_sentence_transformer_config(model_name_or_path: str, **hf_kwargs) -> SbertConfig:
    """Get the configuration of a SentenceTransformers model.

    Args:
    ----
        model_name_or_path (str): Path or name of the model.
        **hf_kwargs: Additional arguments for downloading.

    Returns:
    -------
        SbertConfig: Configuration for the SentenceBERT model.

    Raises:
    ------
        ValueError: If required configuration files are not found.

    """
    try:
        sentence_bert_config_path = download_file_from_hf(
            model_name_or_path, filename="sentence_bert_config.json", **hf_kwargs
        )
        if not sentence_bert_config_path:
            msg = f"sentence_bert_config.json not found for model {model_name_or_path}"
            raise ValueError(msg)

        with open(sentence_bert_config_path) as file:
            sentence_bert_config = json.load(file)
            max_seq_length = sentence_bert_config["max_seq_length"]

        modules_json_path = download_file_from_hf(model_name_or_path, "modules.json", **hf_kwargs)
        if not modules_json_path:
            msg = f"modules.json not found for model {model_name_or_path}"
            raise ValueError(msg)

        with open(modules_json_path) as file:
            modules_config = json.load(file)

        pooling_module = first(
            item for item in modules_config if item["type"] == "sentence_transformers.models.Pooling"
        )
        if not pooling_module:
            msg = "Pooling module not found in modules configuration"
            raise ValueError(msg)

        pooling_config_path = download_file_from_hf(
            model_name_or_path, filename="config.json", subfolder=pooling_module["path"], **hf_kwargs
        )
        if not pooling_config_path:
            msg = f"config.json not found for pooling module in model {model_name_or_path}"
            raise ValueError(msg)

        with open(pooling_config_path) as file:
            pooling_config = json.load(file)

        dimension = pooling_config["word_embedding_dimension"]
        pooling_strategy = _determine_pooling_strategy(pooling_config)
        normalize = any(item["type"] == "sentence_transformers.models.Normalize" for item in modules_config)

        return SbertConfig(
            pooling_strategy=pooling_strategy,
            embedding_dim=dimension,
            max_seq_length=max_seq_length,
            normalize=normalize,
        )
    except Exception as e:
        LOGGER.error(f"Error getting SentenceBERT config: {e}")
        raise


def _determine_pooling_strategy(pooling_config: dict) -> PoolingStrategy:
    """Determine the pooling strategy from the pooling configuration.

    Args:
    ----
        pooling_config (dict): Pooling configuration dictionary.

    Returns:
    -------
        PoolingStrategy: The determined pooling strategy.

    Raises:
    ------
        ValueError: If no valid pooling strategy is found.

    """
    strategy_mapping = {
        "pooling_mode_cls_token": PoolingStrategy.CLS,
        "pooling_mode_mean_tokens": PoolingStrategy.MEAN,
        "pooling_mode_max_tokens": PoolingStrategy.MAX,
        "pooling_mode_mean_sqrt_len_tokens": PoolingStrategy.MEAN_SQRT_LEN,
        "pooling_mode_lasttoken": PoolingStrategy.LAST,
        "pooling_mode_weightedmean_tokens": PoolingStrategy.WEIGHTED_MEAN,
    }

    for config_key, strategy in strategy_mapping.items():
        if pooling_config.get(config_key):
            return strategy

    msg = "No valid pooling strategy found in configuration"
    raise ValueError(msg)


def get_model_config(name_or_path: str, **hf_kwargs) -> dict:
    """Get the model configuration of a Hugging Face model.

    Args:
    ----
        name_or_path (str): Path or name of the model.
        **hf_kwargs: Additional arguments for downloading.

    Returns:
    -------
        dict: Model configuration.

    Raises:
    ------
        ValueError: If config.json is not found.

    """
    try:
        config_path = download_file_from_hf(name_or_path, "config.json", **hf_kwargs)
        if not config_path:
            msg = f"config.json not found for model {name_or_path}"
            raise ValueError(msg)

        with open(config_path) as file:
            return json.load(file)
    except Exception as e:
        raise


def list_hf_onnx_files(repo_id: str) -> list[str]:
    """List all ONNX files in the given repository.

    Args:
    ----
        repo_id (str): Repository ID.

    Returns:
    -------
        List[str]: List of ONNX files.

    """
    return search_repo(repo_id, ".onnx")


def search_repo(repo_id: str, pattern: str, **kwargs) -> list[str]:
    """Search for a pattern in the given repository.

    Args:
    ----
        repo_id (str): Repository ID.
        pattern (str): Pattern to search.
        **kwargs: Additional arguments for searching.

    Returns:
    -------
        List[str]: List of files matching the pattern.

    """
    try:
        result = list_repo_files(repo_id, **kwargs)
        return [file for file in result if pattern in file]
    except Exception as e:
        LOGGER.error(f"Failed to search for {pattern} in {repo_id}: {e}")
        return []


def download_file_from_hf(
    repo_id: str,
    filename: str,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = DEFAULT_CACHE_DIR,
) -> str | None:
    """Download a file from the Hugging Face hub.

    Args:
    ----
        repo_id (str): Repository ID.
        filename (str): Name of the file to download.
        subfolder (Optional[str]): Subfolder path. Defaults to None.
        revision (Optional[str]): Revision of the file. Defaults to None.
        token (Optional[str]): Authentication token. Defaults to None.
        cache_dir (Optional[str]): Cache directory. Defaults to DEFAULT_CACHE_DIR.

    Returns:
    -------
        Optional[str]: Path to the downloaded file or None if download fails.

    """
    path = os.path.join(subfolder, filename) if subfolder else filename
    try:
        return hf_hub_download(
            repo_id,
            filename=path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
    except Exception as e:
        local_path = os.path.join(repo_id, path) if repo_id.startswith("/") else os.path.join(cache_dir, repo_id, path)

        if os.path.exists(local_path):
            return local_path

    return None
