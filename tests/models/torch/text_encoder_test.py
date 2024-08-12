import numpy
import pytest

from ofen.common.utils import ensure_import
from ofen.models.torch.text_encoder import TextEncoder
from ofen.processors.text_processor import TextProcessor

with ensure_import("ofen[torch]"):
    import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def text_encoder():
    processor = TextProcessor(name_or_path=MODEL_NAME, max_length=128)
    return TextEncoder(name_or_path=MODEL_NAME, processor=processor, pooling_strategy="mean")


def test_text_encoder_initialization(text_encoder):
    assert isinstance(text_encoder, TextEncoder)
    assert text_encoder.name_or_path == MODEL_NAME
    assert isinstance(text_encoder.processor, TextProcessor)


def test_text_encoder_pre_process(text_encoder):
    input_text = ["Hello, world!", "This is a test."]
    features = text_encoder.pre_process(input_text)

    assert "input_ids" in features
    assert "attention_mask" in features
    assert features["input_ids"].shape[0] == len(input_text)
    assert features["attention_mask"].shape[0] == len(input_text)


def test_text_encoder_forward(text_encoder):
    input_text = ["Hello, world!", "This is a test."]
    features = text_encoder.pre_process(input_text)
    outputs = text_encoder.forward(**features)

    assert "embeddings" in outputs
    assert outputs["embeddings"].shape[0] == len(input_text)


def test_text_encoder_pipe(text_encoder):
    input_text = ["Hello, world!", "This is a test."]
    outputs = text_encoder.pipe(input_text)

    assert "embeddings" in outputs
    assert isinstance(outputs["embeddings"], torch.Tensor)
    assert outputs["embeddings"].shape[0] == len(input_text)


def test_text_encoder_encode(text_encoder):
    input_text = ["Hello, world!", "This is a test."]
    outputs = text_encoder.encode(input_text)

    assert isinstance(outputs.embeddings, numpy.ndarray)
    assert outputs.embeddings.shape[0] == len(input_text)


def test_text_encoder_encode_batch_size(text_encoder):
    input_text = ["Hello, world!"] * 10
    outputs = text_encoder.encode(input_text, batch_size=4)
    
    assert isinstance(outputs.embeddings, numpy.ndarray)
    assert outputs.embeddings.shape[0] == len(input_text)


@pytest.mark.parametrize("normalize", [True, False])
def test_text_encoder_encode_normalize(text_encoder, normalize):
    input_text = ["Hello, world!", "This is a test."]
    outputs = text_encoder.encode(input_text, normalize=normalize)
    embeddings = outputs.embeddings

    if normalize:
        assert numpy.allclose(numpy.linalg.norm(embeddings, axis=1), numpy.ones(len(input_text)))
    else:
        assert not numpy.allclose(numpy.linalg.norm(embeddings, axis=1), numpy.ones(len(input_text)))


def test_text_encoder_stream_encode(text_encoder):
    input_text = ["Hello, world!", "This is a test.", "Another sentence.", "One more for good measure."]
    batch_size = 2
    
    embeddings = []
    for batch in text_encoder.stream_encode(input_text, batch_size=batch_size):
        batch_embeddings = batch.embeddings
        assert isinstance(batch_embeddings, numpy.ndarray)
        assert batch_embeddings.shape[0] <= batch_size
        embeddings.append(batch_embeddings)
    
    all_embeddings = numpy.concatenate(embeddings)
    assert all_embeddings.shape[0] == len(input_text)


def test_text_encoder_stream_encode_normalize(text_encoder):
    input_text = ["Hello, world!", "This is a test.", "Another sentence.", "One more for good measure."]
    batch_size = 2
    
    for normalize in [True, False]:
        embeddings = []
        for batch in text_encoder.stream_encode(input_text, batch_size=batch_size, normalize=normalize):
            embeddings.append(batch.embeddings)
        
        all_embeddings = numpy.concatenate(embeddings)
        
        if normalize:
            norms = numpy.linalg.norm(all_embeddings, axis=1)
            assert numpy.allclose(norms, 1.0, atol=1e-5)
        else:
            norms = numpy.linalg.norm(all_embeddings, axis=1)
            assert not numpy.allclose(norms, 1.0, atol=1e-5)
