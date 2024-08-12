import pytest

from ofen.common.utils import ensure_import
from ofen.models.torch.cross_encoder import CrossEncoder
from ofen.processors.text_processor import TextProcessor

with ensure_import("ofen[torch]"):
    import torch

MODEL_NAME = "mixedbread-ai/mxbai-rerank-xsmall-v1"


@pytest.fixture
def cross_encoder():
    return CrossEncoder(name_or_path=MODEL_NAME)


def test_cross_encoder_initialization(cross_encoder):
    assert isinstance(cross_encoder, CrossEncoder)
    assert cross_encoder.name_or_path == MODEL_NAME
    assert isinstance(cross_encoder.processor, TextProcessor)


def test_cross_encoder_pre_process(cross_encoder):
    input_texts = ["Document 1", "Document 2"]
    features = cross_encoder.pre_process(input_texts)

    assert "input_ids" in features
    assert "attention_mask" in features
    assert features["input_ids"].shape[0] == len(input_texts)
    assert features["attention_mask"].shape[0] == len(input_texts)


def test_cross_encoder_forward(cross_encoder):
    input_texts = ["Document 1", "Document 2"]
    features = cross_encoder.pre_process(input_texts)
    outputs = cross_encoder.forward(**features)

    assert "scores" in outputs
    assert outputs["scores"].shape[0] == len(input_texts)

def test_cross_encoder_pipe(cross_encoder):
    input_texts = ["Document 1", "Document 2"]
    outputs = cross_encoder.pipe(input_texts)

    assert "scores" in outputs
    assert isinstance(outputs["scores"], torch.Tensor)
    assert outputs["scores"].shape[0] == len(input_texts)


def test_cross_encoder_score(cross_encoder):
    input_texts = ["Document 1", "Document 2"]
    query = "Query 1"
    output = cross_encoder.rerank(query, input_texts)

    assert len(output.results) == len(input_texts)


def test_cross_encoder_score_batch_size(cross_encoder):
    input_texts = ["Document"] * 10
    query = "Query 1"
    output = cross_encoder.rerank(query, input_texts, batch_size=4)
    
    assert len(output.results) == len(input_texts)


