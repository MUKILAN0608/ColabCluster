"""CUDA-only diagnostic. Never imported by the controller."""

import importlib
import time

from common.schemas import GpuTestRequest, GpuTestResponse


class GpuTestError(RuntimeError):
    """CUDA diagnostic could not run."""


def run_gpu_test(worker_id: str, request: GpuTestRequest) -> GpuTestResponse:
    """Time float32 matmul on CUDA device 0 and release tensors on every exit."""
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise GpuTestError("PyTorch is unavailable on this worker") from exc
    a = b = result = None
    try:
        if not torch.cuda.is_available():
            raise GpuTestError("CUDA is unavailable on this worker")
        with torch.cuda.device(0), torch.inference_mode():
            size = request.matrix_size
            a = torch.randn(size, size, device="cuda:0", dtype=torch.float32)
            b = torch.randn(size, size, device="cuda:0", dtype=torch.float32)
            for _ in range(3):
                result = torch.mm(a, b)
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            for _ in range(request.iterations):
                result = torch.mm(a, b)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - start
            return GpuTestResponse(
                worker_id=worker_id, gpu=torch.cuda.get_device_name(0),
                cuda_available=True, cuda_version=str(torch.version.cuda),
                torch_version=str(torch.__version__), matrix_size=size,
                iterations=request.iterations, total_time_seconds=elapsed,
                average_time_ms=elapsed * 1000 / request.iterations,
                gpu_memory_allocated_gb=torch.cuda.memory_allocated(0) / 1024**3,
                result_shape=tuple(result.shape),
            )
    except GpuTestError:
        raise
    except Exception as exc:
        raise GpuTestError(f"CUDA benchmark failed: {exc}") from exc
    finally:
        del a, b, result
