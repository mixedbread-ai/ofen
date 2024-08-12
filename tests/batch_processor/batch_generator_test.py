import asyncio
import pytest
from ofen.batch_processor.batch_generator import BatchGenerator, InferenceItem


@pytest.mark.asyncio
async def test_inference_item():
    item = InferenceItem(
        size=10,
        future=asyncio.Future(),
        content={"input": "test"}
    )
    
    assert not item.done()
    
    result = [1, 2, 3]
    item.complete(result)
    
    assert item.done()
    assert await item.get_result() == result
    
    exception = ValueError("Test exception")
    item = InferenceItem(
        size=5,
        future=asyncio.Future(),
        content={"input": "error"}
    )
    item.set_exception(exception)
    
    with pytest.raises(ValueError):
        await item.get_result()


@pytest.mark.asyncio
async def test_batch_generator():
    batch_gen = BatchGenerator(batch_size=3, timeout=0.1)
    
    # Test empty generator
    async def get_first_batch():
        async for batch in batch_gen.optimal_batches():
            return batch
    
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(get_first_batch(), timeout=0.2)
    
    # Test adding items and generating batches
    items = [
        InferenceItem(
            size=1,
            future=asyncio.Future(),
            content={"input": f"test{i}"}
        )
        for i in range(10)
    ]
    batch_gen.extend(items)
    
    batches = []
    async for batch in batch_gen.optimal_batches():
        batches.append(batch)
        if len(batches) == 4:  # We expect 4 batches (3 full, 1 partial)
            break
    
    assert len(batches) == 4
    assert all(len(batch) == 3 for batch in batches[:3])
    assert len(batches[3]) == 1
    
    # Test prioritized items
    prioritized_item = InferenceItem(
        size=1,
        future=asyncio.Future(),
        content={"input": "priority"},
        prioritized=True
    )
    batch_gen.extend([prioritized_item])
    
    async for batch in batch_gen.optimal_batches():
        assert prioritized_item in batch
        break


@pytest.mark.asyncio
async def test_batch_generator_concurrent():
    batch_gen = BatchGenerator(batch_size=5, timeout=0.1)
    
    async def producer():
        for i in range(20):
            item = InferenceItem(
                size=1,
                future=asyncio.Future(),
                content={"input": f"test{i}"}
            )
            batch_gen.extend([item])
            await asyncio.sleep(0.05)
    
    async def consumer():
        batches = []
        async for batch in batch_gen.optimal_batches():
            batches.append(batch)
            if len(batches) == 4:
                break
        return batches
    
    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    
    batches = await consumer_task
    await producer_task
    
    assert len(batches) == 4
    assert all(1 <= len(batch) <= 5 for batch in batches)
