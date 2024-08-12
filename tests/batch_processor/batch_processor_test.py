import asyncio
from unittest.mock import Mock, AsyncMock

import pytest
# Assuming you're using PyTorch for tensor operations
import torch

from ofen.batch_processor.batch_processor import BatchProcessor, BatchProcessorConfig, ModelInputHelper


@pytest.fixture
def mock_inference_func():
    return AsyncMock()


@pytest.fixture
def batch_processor(mock_inference_func):
    return BatchProcessor(mock_inference_func)


def test_batch_processor_initialization():
    config = BatchProcessorConfig(timeout=0.01, max_batch_size=64)
    inference_func = AsyncMock()
    processor = BatchProcessor.from_config(config, inference_func)
    
    assert processor._config.timeout == 0.01
    assert processor._config.max_batch_size == 64
    assert processor._inference_func == inference_func


@pytest.mark.asyncio
async def test_batch_processor_infer(batch_processor, mock_inference_func):
    mock_inference_func.return_value = {"output": torch.tensor([1, 2, 3])}
    
    result = batch_processor.infer(input=torch.zeros((3, 3))).result()
    
    assert "output" in result
    assert torch.all(result["output"].eq(torch.tensor([1, 2, 3])))
    mock_inference_func.assert_called_once()


def test_model_input_helper_unstack():
    inputs = {"a": torch.tensor([[1, 2], [3, 4]]), "b": torch.tensor([[5, 6], [7, 8]])}
    unstacked = ModelInputHelper.unstack(inputs)
    
    assert len(unstacked) == 2
    assert torch.all(unstacked[0]["a"].eq(torch.tensor([1, 2])))
    assert torch.all(unstacked[0]["b"].eq(torch.tensor([5, 6])))
    assert torch.all(unstacked[1]["a"].eq(torch.tensor([3, 4])))
    assert torch.all(unstacked[1]["b"].eq(torch.tensor([7, 8])))


def test_model_input_helper_stack_results():
    results = [{"output": torch.tensor([1, 2])}, {"output": torch.tensor([3, 4])}]
    stacked = ModelInputHelper.stack_results(results)
    
    assert "output" in stacked
    assert torch.all(stacked["output"].eq(torch.tensor([[1, 2], [3, 4]])))

def test_model_input_helper_pad_and_stack():
    items = [
        Mock(content={"input": torch.tensor([1, 2])}),
        Mock(content={"input": torch.tensor([3, 4, 5])})
    ]
    pad_tokens = {"input": 0}
    
    padded = ModelInputHelper.pad_and_stack(items, pad_tokens)
    
    assert "input" in padded
    assert torch.all(padded["input"].eq(torch.tensor([[1, 2, 0], [3, 4, 5]])))

@pytest.mark.asyncio
async def test_batch_processor_stats(batch_processor, mock_inference_func):
    mock_inference_func.return_value = {"output": torch.tensor([[1, 2], [3, 4]])}
    
    batch_processor.start()
    await asyncio.sleep(0.1)  # Allow time for the processor to start
    
    await asyncio.gather(
        asyncio.wrap_future(batch_processor.infer(input=torch.zeros((2, 3)))),
        asyncio.wrap_future(batch_processor.infer(input=torch.ones((2, 3)))),
    )
    
    stats = batch_processor.stats()
    
    assert stats.total_processed == 4
    assert stats.total_batches == 2
    assert stats.avg_batch_size == 2.0
    assert stats.avg_processing_time > 0
    
    batch_processor.shutdown()
