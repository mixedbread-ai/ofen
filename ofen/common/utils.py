from __future__ import annotations

import asyncio
import functools
import inspect
import pathlib
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Generator, Iterable, Sequence
from contextlib import contextmanager
from queue import Queue
from types import FunctionType, MethodType
from typing import Any, Callable, TypeVar, get_type_hints

from ofen.logger import LOGGER

# Type variables
T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
V = TypeVar("V")
TA = TypeVar("TA")
FA = TypeVar("FA", bound=Callable[..., Any])

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        yield from args[0] if args else ()


@contextmanager
def ensure_import(install_name: str | None = None):
    """Ensure a module is imported, raising a meaningful error if not.

    Args:
    ----
        install_name (Optional[str]): Name of the package to install if import fails.

    Raises:
    ------
        ImportError: If the module cannot be imported.

    """
    try:
        yield
    except ImportError as e:
        module_name = str(e).split("'")[1]
        install_name = install_name or module_name
        msg = (
            f"Failed to import {module_name}. This is required for this feature. "
            f"Please install it using: 'pip install {install_name}'"
        )
        raise ImportError(msg) from e


@contextmanager
def try_import(install_name: str | None = None):
    """Attempt to import a module, logging a message if not available.

    Args:
    ----
        install_name (Optional[str]): Name of the package to suggest for installation.

    """
    try:
        yield
    except ImportError as e:
        module_name = str(e).split("'")[1]
        install_name = install_name or module_name
        LOGGER.info(
            f"Failed to import {module_name}. This is optional for improved functionality. "
            f"Consider installing it using: 'pip install {install_name}'"
        )


def is_method(func: Callable) -> bool:
    """Check if a callable is a method.

    Args:
    ----
        func (Callable): The function to check.

    Returns:
    -------
        bool: True if the callable is a method, False otherwise.

    """
    if not isinstance(func, (FunctionType, MethodType)):
        return False
    return next(iter(inspect.signature(func).parameters)) in ("self", "cls")


def ensure_dir_exists(path: str | pathlib.Path) -> None:
    """Ensure that the directory for the given path exists.

    Args:
    ----
        path (Union[str, pathlib.Path]): The path to check and create if necessary.

    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def batch_iter(seq: Sequence[T], batch_size: int) -> Generator[Sequence[T], None, None]:
    """Yield batches from a sequence.

    Args:
        seq (Sequence[T]): The sequence to batch.
        batch_size (int): The size of each batch.

    Yields:
        Sequence[T]: Batches of the input sequence.

    """
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def first(it: Iterable[T]) -> T:
    """Return the first item from an iterable.

    Args:
        it (Iterable[T]): The iterable to get the first item from.

    Returns:
        T: The first item in the iterable.

    Raises:
        StopIteration: If the iterable is empty.

    """
    return next(iter(it))


def auto_device() -> str:
    """Determine the appropriate device type for PyTorch operations.

    Returns
        str: The name of the device to use ('cuda', 'npu', 'mps', or 'cpu').

    """
    with try_import():
        import torch
        from transformers import is_torch_npu_available

        if torch.cuda.is_available():
            return "cuda"
        if is_torch_npu_available():
            return "npu"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def bucket_batch_iter(
    data: list[T], batch_size: int, descending: bool = True
) -> Generator[tuple[list[T], list[int]], None, None]:
    """Iterate over data in batches, sorted by length.

    This function sorts the input data by length, then yields batches of the specified size.
    It's useful for processing sequences of varying lengths efficiently.

    Args:
    ----
        data (List[T]): List of items to batch.
        batch_size (int): Number of items per batch.
        descending (bool): If True, sort by length in descending order. Defaults to True.

    Yields:
    ------
        Tuple[List[T], List[int]]: A tuple containing a batch of items and their original indices.

    """
    if len(data) <= batch_size:
        yield data, list(range(len(data)))
        return

    data_with_indices = sorted(enumerate(data), key=lambda x: len(x[1]), reverse=descending)
    for batch in batch_iter(data_with_indices, batch_size):
        indices, items = zip(*batch)
        yield list(items), list(indices)


def threaded_generator(
    compute_fn: Callable[[T], U], generator_or_list: Iterable[T] | list[T], max_queue_size: int = 20
) -> Generator[U, None, None]:
    """Generate results using a separate thread.

    This function processes items from the input generator or list in a separate thread,
    allowing for concurrent computation and yielding.

    Args:
    ----
        compute_fn (Callable[[T], U]): Function to apply to each item.
        generator_or_list (Union[Iterable[T], List[T]]): Input data to process.
        max_queue_size (int): Maximum number of computed results to keep in queue. Defaults to 20.

    Yields:
    ------
        U: Computed results from compute_fn.

    """
    queue: Queue[U | type] = Queue(maxsize=max_queue_size)
    stop_event = threading.Event()

    def worker():
        try:
            for item in generator_or_list:
                if stop_event.is_set():
                    break
                queue.put(compute_fn(item))
        except Exception as e:
            LOGGER.error(f"Error in threaded generator: {e!s}")
            queue.put(e)
        finally:
            queue.put(StopIteration)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            item = queue.get()  # Add a small timeout to allow for checking stop_event
            if item is StopIteration:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    except Exception as e:
        print(e)
    finally:
        stop_event.set()
        thread.join()
        queue.queue.clear()


def add_async_methods(postfix: str = "_async"):
    """Class decorator to add asynchronous versions of methods.

    This decorator adds an asynchronous version of each method in the class,
    with the specified postfix added to the method name.

    Args:
    ----
        postfix (str): String to append to method names for async versions. Defaults to '_async'.

    Returns:
    -------
        Callable[[type[TA]], type[TA]]: Decorated class with additional async methods.

    """

    def decorator(cls: type[TA]) -> type[TA]:
        def wrap_async(func: FA) -> FA:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(func, *args, **kwargs)

            wrapper.__annotations__ = {k: Awaitable[v] if k == "return" else v for k, v in get_type_hints(func).items()}
            return wrapper

        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("__") and not inspect.iscoroutinefunction(method):
                setattr(cls, f"{name}{postfix}", wrap_async(method))

        return cls

    return decorator


def identity(x: Any) -> Any:
    """
    Identity function that returns the input unchanged.

    This function is useful in scenarios where a function is required as an argument,
    but no transformation of the input is needed.

    Args:
        x (Any): The input value of any type.

    Returns:
        Any: The same value that was passed as input.
    """
    return x


class Timer:
    """Utility class for timing code execution of arbitrary actions."""

    def __init__(self):
        self.total_times = {}
        self.counts = {}
        self._start_time = time.perf_counter()
        self._end_time = None

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure the execution time of a code block.

        Args:
            name (str): The name of the action being timed.
        """
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        self.total_times[name] = self.total_times.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def total(self, name: str) -> float:
        """Get the total time spent on a specific action.

        Args:
            name (str): The name of the action.

        Returns:
            float: Total time spent on the action.
        """
        return self.total_times.get(name, 0.0)

    def average(self, name: str) -> float:
        """Get the average time per occurrence of a specific action.

        Args:
            name (str): The name of the action.

        Returns:
            float: Average time per occurrence.
        """
        count = self.counts.get(name, 0)
        if count == 0:
            return 0.0
        return self.total_times[name] / count

    def total_elapsed(self) -> float:
        """Get the total elapsed time since the timer was started.

        Returns:
            float: Total elapsed time.
        """
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time

    def end(self):
        """Mark the end time of the timer."""
        self._end_time = time.perf_counter()

    def to_dict(self) -> dict:
        """Convert the timer data to a dictionary.

        Returns:
            dict: A dictionary containing timer data.
        """
        result = defaultdict(float)
        for key in self.total_times:
            result[key] = self.total(key)
            result[f"avg_{key}"] = self.average(key)
        return result
