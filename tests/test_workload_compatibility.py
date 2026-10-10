"""Controller/worker OpenAPI and validation agree, without live execution."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from common.schemas import WorkerStatus
from common.scaling import VALIDATION_SIZES
from controller.main import create_app
from worker.diagnostics import create_diagnostic_app
from scripts import validate_scaling as validation
from test_scaling import worker_result

ALL_SIZES = [64,256,512,768,1024,1536,2048,3072,4096]


def test_actual_controller_and_worker_schemas():
    controller=create_app().openapi()
    worker=create_diagnostic_app(SimpleNamespace(worker_id="a",status=WorkerStatus.READY)).openapi()
    for schema,name in [(controller,"ScalingRequest"),(worker,"ScalingBatchRequest")]:
        ref=schema["paths"]["/inference/scaling"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("/"+name)
        assert schema["components"]["schemas"][name]["properties"]["total_samples"]["enum"]==ALL_SIZES


@pytest.mark.parametrize("size",ALL_SIZES)
def test_supported_sizes_reach_both_handlers(monkeypatch,size):
    execute=Mock(side_effect=RuntimeError("controller validation passed"))
    monkeypatch.setattr("controller.api.run_scaling",execute)
    # Catch the sentinel after FastAPI validation; no orchestration or GPU call.
    with TestClient(create_app()) as client:
        with pytest.raises(RuntimeError,match="controller validation passed"):
            client.post("/inference/scaling",json={"total_samples":size,"worker_count":1})
    assert execute.call_args.args[1].total_samples==size
    gpu=Mock(return_value=worker_result(size,1,0))
    monkeypatch.setattr("worker.diagnostics.run_inference",gpu)
    with TestClient(create_diagnostic_app(SimpleNamespace(worker_id="COLAB-GPU-TEST",status=WorkerStatus.READY))) as client:
        assert client.post("/inference/scaling",json={"total_samples":size,"worker_count":1,"partition":0}).status_code==200
    assert gpu.call_args.kwargs["scaling"].assigned_samples==size


@pytest.mark.parametrize("size",[0,63,65,128,1000,4097,"1024",None])
def test_unsupported_sizes_rejected_before_execution(monkeypatch,size):
    dispatch=Mock();gpu=Mock()
    monkeypatch.setattr("controller.api.run_scaling",dispatch)
    monkeypatch.setattr("worker.diagnostics.run_inference",gpu)
    with TestClient(create_app()) as controller, TestClient(create_diagnostic_app(SimpleNamespace(worker_id="a",status=WorkerStatus.READY))) as worker:
        body={"total_samples":size,"worker_count":1}
        assert controller.post("/inference/scaling",json=body).status_code==422
        assert worker.post("/inference/scaling",json={**body,"partition":0}).status_code==422
    dispatch.assert_not_called();gpu.assert_not_called()


@pytest.mark.parametrize("outdated",["controller","worker",None])
def test_preflight_diagnoses_actual_missing_sizes(monkeypatch,outdated):
    from common.schemas import WorkerInfo
    from datetime import datetime,timezone
    now=datetime.now(timezone.utc)
    records={name:WorkerInfo(worker_id=name,gpu="Tesla T4",gpu_memory=14.56,cuda_available=True,
        status="ready",registered_at=now,last_seen=now,metadata={"diagnostic_url":f"https://w{i}.example"}).model_dump()
        for i,name in enumerate(validation.WORKER_IDS)}
    monkeypatch.setattr(validation.scaling,"preflight",Mock(return_value=records))
    def get(url):
        is_controller=url.startswith("http://controller")
        name="ScalingRequest" if is_controller else "ScalingBatchRequest"
        stale=(is_controller and outdated=="controller") or (not is_controller and outdated=="worker")
        values=[64,256,1024,4096] if stale else ALL_SIZES
        return {"components":{"schemas":{name:{"properties":{"total_samples":{"enum":values}}}}}}
    monkeypatch.setattr(validation.scaling,"get_json",get)
    if outdated:
        with pytest.raises(ValueError) as error:validation.preflight("http://controller",VALIDATION_SIZES)
        assert "missing=[512, 768, 1536, 2048, 3072]" in str(error.value)
        assert ("Stop the existing controller" if outdated=="controller" else "Upload the current worker bundle") in str(error.value)
    else:
        assert validation.preflight("http://controller",VALIDATION_SIZES)==records
