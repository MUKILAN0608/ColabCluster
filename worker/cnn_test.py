"""Fixed SmallCNN inference on the remote CUDA worker; no training or datasets."""

import importlib
import time
from typing import Any

from common.schemas import CnnTestResponse


class CnnTestError(RuntimeError):
    """The CUDA worker could not complete SmallCNN inference."""


def build_small_cnn(torch: Any) -> Any:
    """Build the fixed architecture after the optional PyTorch import succeeds."""
    class SmallCNN(torch.nn.Module):
        """Two convolution/pooling stages followed by a ten-class output head."""

        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
                torch.nn.Linear(32, 10),
            )

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    return SmallCNN()


def run_cnn_test(worker_id: str) -> CnnTestResponse:
    """Run ten warmups and 100 timed float32 forwards, returning real GPU metrics."""
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise CnnTestError("PyTorch is unavailable on this worker") from exc
    model = inputs = output = generator = None
    try:
        if not torch.cuda.is_available():
            raise CnnTestError("CUDA is unavailable; the CNN diagnostic requires a CUDA worker")
        # Preserve CPU RNG state for model initialization; use a private CUDA RNG
        # for synthetic input so unrelated worker random streams are not reseeded.
        with torch.cuda.device(0), torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            model = build_small_cnn(torch).to(device="cuda:0", dtype=torch.float32)
            model.eval()
            generator = torch.Generator(device="cuda:0").manual_seed(0)
            inputs = torch.randn(32, 3, 32, 32, device="cuda:0", dtype=torch.float32,
                                 generator=generator)
            for _ in range(10):
                output = model(inputs)
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            start = time.perf_counter()
            for _ in range(100):
                output = model(inputs)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - start
            if elapsed <= 0:
                raise CnnTestError("CUDA timing did not produce a positive duration")
            return CnnTestResponse(
                worker_id=worker_id, gpu=torch.cuda.get_device_name(0),
                cuda_version=str(torch.version.cuda), torch_version=str(torch.__version__),
                device=str(output.device), model="SmallCNN", input_shape=tuple(inputs.shape),
                batch_size=32, forward_passes=100, total_gpu_time_ms=elapsed * 1000,
                average_inference_ms=elapsed * 1000 / 100,
                throughput_images_per_second=32 * 100 / elapsed,
                peak_memory_mb=torch.cuda.max_memory_allocated(0) / 1024**2,
                output_shape=tuple(output.shape), status="passed",
            )
    except CnnTestError:
        raise
    except Exception as exc:
        raise CnnTestError(f"SmallCNN CUDA diagnostic failed: {exc}") from exc
    finally:
        del model, inputs, output, generator
