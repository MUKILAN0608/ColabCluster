"""Bounded concurrent dispatch for exactly two fixed synthetic batches."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import requests
from controller.transport import post_execution
from fastapi import HTTPException
from pydantic import HttpUrl, TypeAdapter, ValidationError

from common.schemas import InferenceBatchResponse, TwoWorkerResponse, SingleInferenceResponse, SingleWorkerResponse
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


def forward(worker, url, partition, *, single=False):
    payload = {} if single else {"partition": partition}
    if single:
        start = perf_counter()
    response = post_execution(worker, url, payload)
    if single:
        elapsed = perf_counter() - start
    try:
        try:
            response_type = SingleInferenceResponse if single else InferenceBatchResponse
            result = response_type.model_validate(response.json())
            if (result.worker_id != worker.worker_id or (not single and result.partition != partition)
                    or result.cuda_version in ("", "None") or not result.torch_version):
                raise ValueError()
        except ValueError:
            raise HTTPException(502, f"Invalid inference result from {worker.worker_id}") from None
        if single:
            if elapsed <= 0:
                raise HTTPException(502, "Wall-clock measurement was not positive")
            return SingleWorkerResponse(**result.model_dump(), wall_time_ms=elapsed * 1000,
                                        effective_throughput_images_per_second=64 / elapsed)
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
            registry.finish_nn_test(worker.worker_id, original, worker)


def run_single_worker(registry, timeout):
    try:
        worker, original = registry.begin_single_worker_inference(timeout)
    except WorkerNotReadyError:
        raise HTTPException(409, "No READY workers available") from None
    try:
        url = diagnostic_url(worker) + "/single-worker"
        return forward(worker, url, 0, single=True)
    finally:
        registry.finish_nn_test(worker.worker_id, original, worker)
