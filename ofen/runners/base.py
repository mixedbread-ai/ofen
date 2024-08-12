from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofen.batch_processor.batch_processor import BatchProcessorStats


@dataclasses.dataclass
class Usage:
    """Represents token usage information for a model request.

    Attributes
    ----------
        total_tokens (int): The total number of tokens used in the request.
        prompt_tokens (int): The number of tokens used in the prompt. Defaults to 0.
        completion_tokens (int): The number of tokens used in the completion. Defaults to 0.

    """

    total_tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclasses.dataclass
class RunnerRequestInfo:
    """Tracks information about a runner request, including timing and usage statistics.

    Attributes
    ----------
        arrival_time (float): The time when the request arrived. Defaults to current time.
        end_time (float): The time when the request completed. Initialized to -1.
        duration (float): The total duration of the request. Initialized to -1.
        usage (Optional[Usage]): Token usage information for the request. Defaults to None.
        batch_stats_in (BatchProcessorStats): Input batch processing statistics. Defaults to None.
        batch_stats_out (BatchProcessorStats): Output batch processing statistics. Defaults to None.

    """

    arrival_time: float = dataclasses.field(default_factory=time.time)
    end_time: float = dataclasses.field(default=-1.0, init=False)
    duration: float = dataclasses.field(default=-1.0, init=False)
    usage: Usage | None = None
    batch_stats_in: BatchProcessorStats | None = None
    batch_stats_out: BatchProcessorStats | None = None

    def __enter__(self) -> RunnerRequestInfo:
        """Context manager entry method. Sets the arrival time to the current time.

        Returns
        -------
            RunnerRequestInfo: The current instance.

        """
        self.arrival_time = time.time()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object | None) -> None:
        """Context manager exit method. Calculates the request duration.

        Args:
        ----
            exc_type (Optional[type]): The type of the exception that was raised, if any.
            exc_val (Optional[Exception]): The exception instance that was raised, if any.
            exc_tb (Optional[object]): The traceback object encapsulating the call stack at the point where the exception occurred.

        """
        self.end_time = time.time()
        self.duration = self.end_time - self.arrival_time
