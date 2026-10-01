"""Inspect local resources without requiring PyTorch or a GPU."""

import importlib
import logging
import math
import platform

import psutil

from common.schemas import HardwareInfo

BYTES_PER_GIB = 1024 ** 3


def get_gpu_metrics(cuda_available: bool) -> tuple[float | None, float | None]:
    """Best-effort utilization percent and global used VRAM in GiB for device 0."""
    if not cuda_available:
        return None, None
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None, None
    utilization = None
    memory_used = None
    try:
        value = float(torch.cuda.utilization(0))
        if math.isfinite(value) and 0 <= value <= 100:
            utilization = value
    except Exception:
        pass  # Optional NVML support may be absent even when CUDA is available.
    try:
        free, total = torch.cuda.mem_get_info(0)
        value = (float(total) - float(free)) / BYTES_PER_GIB
        if 0 <= free <= total and math.isfinite(value):
            memory_used = value
    except Exception:
        pass  # A metrics failure must not prevent the heartbeat itself.
    return utilization, memory_used


def _detect_gpu() -> tuple[str, float, bool, str | None]:
    """Probe the first CUDA device; optional dependency failures are recoverable."""
    torch_version = None
    try:
        torch = importlib.import_module("torch")
        torch_version = str(torch.__version__)
        if torch.cuda.is_available():
            name = str(torch.cuda.get_device_name(0)).strip()
            memory = float(torch.cuda.get_device_properties(0).total_memory)
            if not name or not 0 < memory < float("inf"):
                raise ValueError("CUDA returned invalid device properties")
            return name, memory / BYTES_PER_GIB, True, torch_version
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            logging.getLogger(__name__).warning("PyTorch dependency unavailable: %s", exc)
    except Exception as exc:
        # Optional native libraries can fail on import or during driver initialization.
        logging.getLogger(__name__).warning("GPU detection unavailable: %s", exc)
    return "None", 0.0, False, torch_version


def get_hardware_info() -> HardwareInfo:
    """Return a validated snapshot with GiB memory units and logical CPU count."""
    cpu_count = max(1, psutil.cpu_count(logical=True) or 1)
    ram_total = psutil.virtual_memory().total / BYTES_PER_GIB
    gpu, gpu_memory, cuda_available, torch_version = _detect_gpu()
    return HardwareInfo(
        cpu_count=cpu_count,
        ram_total=ram_total,
        gpu=gpu,
        gpu_memory=gpu_memory,
        cuda_available=cuda_available,
        torch_version=torch_version,
        python_version=platform.python_version(),
        platform=platform.system(),
    )
