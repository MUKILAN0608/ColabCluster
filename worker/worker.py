"""Local worker entry point: python -m worker.worker."""

import os
from uuid import uuid4

from common.schemas import HardwareInfo, LocalWorkerInfo
from worker.hardware import get_hardware_info

# An immutable fallback identity, generated once per interpreter process.
_PROCESS_WORKER_ID = f"worker-{uuid4().hex}"


def get_worker_id() -> str:
    """Use a nonblank environment ID or the process-stable UUID fallback."""
    configured = os.environ.get("COLABCLUSTER_WORKER_ID", "").strip()
    return configured or _PROCESS_WORKER_ID


class Worker:
    """One local compute runtime with an immutable initialization snapshot."""

    def __init__(self, worker_id: str | None = None) -> None:
        identifier = get_worker_id() if worker_id is None else worker_id.strip()
        if not identifier:
            raise ValueError("worker_id cannot be empty")
        self._info = LocalWorkerInfo(worker_id=identifier, hardware=get_hardware_info())

    @property
    def worker_id(self) -> str:
        """Return this instance's fixed identity."""
        return self._info.worker_id

    def get_hardware_info(self) -> HardwareInfo:
        """Return the immutable hardware snapshot captured at initialization."""
        return self._info.hardware

    def get_info(self) -> LocalWorkerInfo:
        """Return structured local identity and hardware information."""
        return self._info


def main() -> None:
    """Print actual local hardware information and exit without networking."""
    worker = Worker()
    hardware = worker.get_hardware_info()
    print("=" * 40)
    print("        COLABCLUSTER WORKER")
    print("=" * 40)
    for label, value in (
        ("Worker ID", worker.worker_id),
        ("CPU Cores", hardware.cpu_count),
        ("RAM", f"{hardware.ram_total:.2f} GiB"),
        ("GPU", hardware.gpu),
        ("GPU Memory", f"{hardware.gpu_memory:.2f} GiB"),
        ("CUDA Available", hardware.cuda_available),
        ("PyTorch", hardware.torch_version or "Not installed or unavailable"),
        ("Python", hardware.python_version),
        ("Platform", hardware.platform),
    ):
        print(f"{label:<15}: {value}")
    print("=" * 40)


if __name__ == "__main__":
    main()
