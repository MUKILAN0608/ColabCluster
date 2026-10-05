"""One fixed, synthetic CUDA inference diagnostic. No training or downloads."""

import importlib
import time

from common.schemas import NnTestResponse


class NnTestError(RuntimeError):
    """The worker could not complete the fixed CUDA diagnostic."""


def run_nn_test(worker_id: str) -> NnTestResponse:
    """Run 100 SmallMLP forward passes with batch 128; release local tensors."""
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise NnTestError("PyTorch is unavailable on this worker") from exc
    model = inputs = output = None
    try:
        if not torch.cuda.is_available():
            raise NnTestError("CUDA is unavailable; this diagnostic requires a CUDA worker")
        with torch.cuda.device(0), torch.inference_mode():
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            model = torch.nn.Sequential(
                torch.nn.Linear(784, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
            ).to(device="cuda:0", dtype=torch.float32)
            model.eval()
            inputs = torch.randn(128, 784, device="cuda:0", dtype=torch.float32)
            for _ in range(3):
                output = model(inputs)
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            for _ in range(100):
                output = model(inputs)
            torch.cuda.synchronize(0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return NnTestResponse(
                worker_id=worker_id, gpu=torch.cuda.get_device_name(0),
                cuda_version=str(torch.version.cuda), torch_version=str(torch.__version__),
                device=str(output.device), model="SmallMLP", batch_size=128, forward_passes=100,
                status="passed", input_shape=tuple(inputs.shape),
                output_shape=tuple(output.shape), total_gpu_time_ms=elapsed_ms,
                average_inference_ms=elapsed_ms / 100,
                peak_memory_mb=torch.cuda.max_memory_allocated(0) / 1024**2,
            )
    except NnTestError:
        raise
    except Exception as exc:
        raise NnTestError(f"SmallMLP CUDA diagnostic failed: {exc}") from exc
    finally:
        del model, inputs, output
