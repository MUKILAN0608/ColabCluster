"""Deterministic local runtime tests that do not require PyTorch or a GPU."""

from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
from pydantic import ValidationError

from common.schemas import HardwareInfo, LocalWorkerInfo
from worker import hardware
from worker.worker import Worker, get_worker_id, main


@pytest.fixture(autouse=True)
def local_hardware(monkeypatch):
    """Supply deterministic CPU/RAM and simulate an absent optional PyTorch."""
    monkeypatch.delenv("COLABCLUSTER_WORKER_ID", raising=False)
    cpu = Mock(return_value=4)
    monkeypatch.setattr(hardware.psutil, "cpu_count", cpu)
    monkeypatch.setattr(hardware.psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 1024 ** 3))
    importer = Mock(side_effect=ModuleNotFoundError("No module named torch", name="torch"))
    monkeypatch.setattr(hardware.importlib, "import_module", importer)
    return cpu, importer


def test_worker_initialization() -> None:
    worker = Worker()
    assert worker.worker_id.startswith("worker-")
    assert isinstance(worker.get_hardware_info(), HardwareInfo)


def test_generated_id_is_stable() -> None:
    identifier = get_worker_id()
    assert UUID(identifier.removeprefix("worker-")).version == 4
    assert get_worker_id() == identifier
    assert Worker().worker_id == Worker().worker_id == identifier


def test_environment_id(monkeypatch) -> None:
    monkeypatch.setenv("COLABCLUSTER_WORKER_ID", " TEST-WORKER-01 ")
    worker = Worker()
    assert get_worker_id() == worker.worker_id == "TEST-WORKER-01"
    monkeypatch.setenv("COLABCLUSTER_WORKER_ID", "CHANGED")
    assert worker.worker_id == "TEST-WORKER-01"


def test_explicit_id_has_precedence(monkeypatch) -> None:
    monkeypatch.setenv("COLABCLUSTER_WORKER_ID", "ENV-ID")
    assert Worker(" explicit ").worker_id == "explicit"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_explicit_id_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Worker(value)


def test_empty_environment_uses_generated_id(monkeypatch) -> None:
    original = get_worker_id()
    monkeypatch.setenv("COLABCLUSTER_WORKER_ID", " ")
    assert get_worker_id() == original


def test_hardware_fields_and_units(local_hardware) -> None:
    info = hardware.get_hardware_info()
    assert set(info.model_dump()) == {
        "cpu_count", "ram_total", "gpu", "gpu_memory", "cuda_available",
        "torch_version", "python_version", "platform",
    }
    assert info.cpu_count == 4
    assert info.ram_total == 16
    assert isinstance(info.cuda_available, bool)
    assert info.python_version and info.platform
    local_hardware[0].assert_called_once_with(logical=True)


@pytest.mark.parametrize("count", [None, 0])
def test_cpu_count_fallback(local_hardware, count) -> None:
    local_hardware[0].return_value = count
    assert hardware.get_hardware_info().cpu_count == 1


def test_missing_torch_fallback() -> None:
    info = hardware.get_hardware_info()
    assert (info.gpu, info.gpu_memory, info.cuda_available, info.torch_version) == ("None", 0, False, None)


def mock_torch(local_hardware, available: bool = True):
    """Install a fake optional PyTorch module for the hardware probe."""
    torch = SimpleNamespace(__version__="test-version", cuda=Mock())
    torch.cuda.is_available.return_value = available
    torch.cuda.get_device_name.return_value = "Simulated GPU"
    torch.cuda.get_device_properties.return_value = SimpleNamespace(total_memory=24 * 1024 ** 3)
    local_hardware[1].side_effect = None
    local_hardware[1].return_value = torch
    return torch


def test_cuda_unavailable(local_hardware) -> None:
    torch = mock_torch(local_hardware, available=False)
    info = hardware.get_hardware_info()
    assert (info.gpu, info.gpu_memory, info.cuda_available) == ("None", 0, False)
    assert info.torch_version == "test-version"
    torch.cuda.get_device_name.assert_not_called()


def test_first_cuda_device(local_hardware) -> None:
    torch = mock_torch(local_hardware)
    info = hardware.get_hardware_info()
    assert (info.gpu, info.gpu_memory, info.cuda_available) == ("Simulated GPU", 24, True)
    torch.cuda.get_device_name.assert_called_once_with(0)
    torch.cuda.get_device_properties.assert_called_once_with(0)


@pytest.mark.parametrize("operation", ["is_available", "get_device_name", "get_device_properties"])
def test_cuda_driver_failures(local_hardware, operation: str) -> None:
    torch = mock_torch(local_hardware)
    getattr(torch.cuda, operation).side_effect = RuntimeError("driver failed")
    info = hardware.get_hardware_info()
    assert (info.gpu, info.gpu_memory, info.cuda_available) == ("None", 0, False)
    assert info.torch_version == "test-version"


@pytest.mark.parametrize("error", [OSError("native library missing"), ImportError("broken install")])
def test_torch_import_failure(local_hardware, error) -> None:
    local_hardware[1].side_effect = error
    assert hardware.get_hardware_info().cuda_available is False


def test_structured_snapshot() -> None:
    worker = Worker("snapshot")
    info = worker.get_info()
    assert isinstance(info, LocalWorkerInfo)
    assert info.worker_id == "snapshot"
    assert LocalWorkerInfo.model_validate_json(info.model_dump_json()) == info
    with pytest.raises(ValidationError):
        info.hardware.gpu = "mutated"


def test_cli_output(capsys, monkeypatch) -> None:
    monkeypatch.setenv("COLABCLUSTER_WORKER_ID", "CLI-TEST")
    main()
    output = capsys.readouterr().out
    for value in ["COLABCLUSTER WORKER", "CLI-TEST", "16.00 GiB", "None", "False", "Python", "Platform"]:
        assert value in output
