"""Bounded concurrent dispatch for exactly two fixed synthetic batches."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import requests
from fastapi import HTTPException
from pydantic import HttpUrl, TypeAdapter, ValidationError

from common.schemas import InferenceBatchResponse, TwoWorkerResponse
from controller.registry import UnknownWorkerError, WorkerNotReadyError


def diagnostic_url(worker):
    try:
        url = TypeAdapter(HttpUrl).validate_python((worker.metadata or {}).get("diagnostic_url"))
        if url.username or url.password or url.query or url.fragment:
            raise ValueError()
        if worker.gpu != "Tesla T4" or not worker.cuda_available:
            raise ValueError()
        return str(url).rstrip("/") + "/inference"
    except (ValueError, ValidationError):
        raise HTTPException(409, f"{worker.worker_id} requires a Tesla T4, CUDA and valid diagnostic URL") from None


def forward(worker, url, partition):
    try:
        response = requests.post(url, json={"partition": partition},
                                 timeout=(5, 120), allow_redirects=False)
    except requests.Timeout:
        raise HTTPException(504, f"{worker.worker_id} timed out; remote work may still be running") from None
    except requests.RequestException:
        raise HTTPException(502, f"Cannot reach {worker.worker_id}") from None
    try:
        if response.status_code != 200:
            raise HTTPException(response.status_code if response.status_code in (409, 503) else 502,
                                f"Inference failed on {worker.worker_id}")
        try:
            result = InferenceBatchResponse.model_validate(response.json())
            if (result.worker_id != worker.worker_id or result.partition != partition
                    or result.cuda_version in ("", "None") or not result.torch_version):
                raise ValueError()
        except ValueError:
            raise HTTPException(502, f"Invalid inference result from {worker.worker_id}") from None
        return result
    finally:
        response.close()


def run_two_worker(registry, worker_ids, timeout):
    try:
        reserved = registry.begin_two_worker_inference(worker_ids, timeout)
    except UnknownWorkerError as exc:
        raise HTTPException(404, f"Worker not active: {exc}") from None
    except WorkerNotReadyError as exc:
        raise HTTPException(409, f"Worker not ready: {exc}") from None
    try:
        urls = [diagnostic_url(worker) for worker, _ in reserved]
        if urls[0] == urls[1]:
            raise HTTPException(409, "Workers must have independent diagnostic URLs")
        start = perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(forward, worker, url, partition)
                       for partition, ((worker, _), url) in enumerate(zip(reserved, urls))]
            results = [future.result() for future in futures]
        elapsed = perf_counter() - start
        return TwoWorkerResponse(
            workers=tuple(results), parallel_wall_time_ms=elapsed * 1000,
            sum_worker_gpu_time_ms=sum(r.total_gpu_time_ms for r in results),
            effective_throughput_images_per_second=64 / elapsed,
            output_shapes=tuple(r.output_shape for r in results))
    finally:
        # Executor shutdown waits for both calls even when one fails.
        for worker, original in reserved:
            registry.finish_nn_test(worker.worker_id, original)
