from typing import Any

import torch
import torch.multiprocessing as mp

from ofen.logger import LOGGER


class TorchModule(torch.nn.Module):
    """A base class for PyTorch modules with additional utility methods, including multi-processing support."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy", torch.empty(0), persistent=False)
        self._mp_pool = None

    @property
    def device(self) -> torch.device:
        """Get the current device of the module."""
        return self._dummy.device

    def to(self, device=None, *args, **kwargs) -> "TorchModule":
        """Move the module to the specified device or devices."""
        self.stop_multi_process_pool()
        if isinstance(device, (list, tuple)):
            # Multiple devices specified
            self.to("cpu")  # Move module to CPU
            self.share_memory()
            self.start_multi_process_pool(target_devices=device)
        else:
            super().to(device, *args, **kwargs)
        return self

    def cuda(self, device_id: int | None = None) -> "TorchModule":
        """Move the module to a CUDA device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return self.to(torch.device("cuda", device_id))

    def cpu(self) -> "TorchModule":
        """Move the module to the CPU."""
        return self.to(torch.device("cpu"))

    def int8(self) -> "TorchModule":
        """Convert the module to INT8."""
        if torch.backends.quantized.engine == "none":
            torch.backends.quantized.engine = "qnnpack"
        return torch.quantization.quantize_dynamic(self, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)

    def start_multi_process_pool(self, target_devices: list[torch.device] | None = None) -> None:
        """
        Starts a multi-process pool to process data with several independent processes.
        The pool is stored internally in the module.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            else:
                LOGGER.info("CUDA is not available. Starting CPU workers.")
                target_devices = [torch.device("cpu")] * mp.cpu_count()

        LOGGER.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device in target_devices:
            p = ctx.Process(
                target=self._multi_process_worker,
                args=(str(device), self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        self._mp_pool = {"input": input_queue, "output": output_queue, "processes": processes}
        self._old_forward = self.forward
        self.forward = self._forward

    def stop_multi_process_pool(self) -> None:
        """
        Stops all processes started with start_multi_process_pool.
        """
        if self._mp_pool is None:
            return  # No pool to stop

        self.forward = self._old_forward
        del self._old_forward

        # Send stop signals to workers
        for _ in self._mp_pool["processes"]:
            self._mp_pool["input"].put((None, None))

        for p in self._mp_pool["processes"]:
            p.join()
            p.close()

        self._mp_pool["input"].close()
        self._mp_pool["output"].close()
        self._mp_pool = None

    def _forward(self, **kwargs) -> Any:
        """
        Processes data using multiple processes if multi-processing is enabled,
        otherwise processes the data in the current process.
        """
        if self._mp_pool is None:
            return super().forward(**kwargs)

        input_queue = self._mp_pool["input"]
        output_queue = self._mp_pool["output"]
        num_workers = len(self._mp_pool["processes"])

        # Assume that all tensors in kwargs have the same batch size in dimension 0
        batch_size = next(iter(kwargs.values())).shape[0]

        # Split kwargs into num_workers chunks
        chunk_sizes = [batch_size // num_workers] * num_workers
        for i in range(batch_size % num_workers):
            chunk_sizes[i] += 1  # Distribute the remainder

        indices = []
        start = 0
        for size in chunk_sizes:
            end = start + size
            indices.append((start, end))
            start = end

        # Prepare kwargs for each worker
        for i, (start, end) in enumerate(indices):
            worker_kwargs = {k: v[start:end] for k, v in kwargs.items()}
            input_queue.put((i, worker_kwargs))

        # Collect results
        results = [None] * num_workers
        for _ in range(num_workers):
            idx, output = output_queue.get()
            results[idx] = output

        # Concatenate results along the batch dimension
        if isinstance(results[0], torch.Tensor):
            output = torch.cat(results, dim=0)
        elif isinstance(results[0], dict):
            output = {}
            for key in results[0].keys():
                output[key] = torch.cat([res[key] for res in results], dim=0)
        else:
            raise TypeError("Output type not supported for concatenation")

        return output

    @staticmethod
    def _multi_process_worker(
        target_device: str, model: "TorchModule", input_queue: mp.Queue, output_queue: mp.Queue
    ) -> None:
        """
        Internal worker process to process data in a multi-process setup.
        """
        model = model.to(target_device)
        while True:
            try:
                idx, worker_kwargs = input_queue.get()
                if idx is None:
                    break  # Stop signal
                # Move input tensors to target device
                worker_kwargs = {k: v.to(target_device, non_blocking=True) for k, v in worker_kwargs.items()}
                with torch.no_grad():
                    output = model.forward(**worker_kwargs)
                # No need to move output to CPU; torch.multiprocessing.Queue can handle CUDA tensors
                output_queue.put((idx, output))
            except Exception:
                LOGGER.exception("Exception in multi_process_worker")
                break

    def __del__(self):
        self.stop_multi_process_pool()
