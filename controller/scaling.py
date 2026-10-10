"""One/two-worker bounded scaling experiment using existing reservations."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import requests
from controller.transport import post_execution
from fastapi import HTTPException
from common.scaling import ScalingBatchRequest, ScalingBatchResponse, ScalingResponse, WORKER_IDS
from controller.inference import diagnostic_url
from controller.registry import UnknownWorkerError, WorkerNotReadyError


def forward_scaling(worker, url, batch):
    response = post_execution(worker, url, batch.model_dump())
    try:
        try:
            result = ScalingBatchResponse.model_validate(response.json())
            if (result.worker_id != worker.worker_id or result.partition != batch.partition
                    or result.batch_size != batch.assigned_samples
                    or result.cuda_version in ("", "None") or not result.torch_version):
                raise ValueError("Invalid identity, CUDA version or assigned sample count")
            return result
        except ValueError:
            raise HTTPException(502, f"Invalid scaling result from {worker.worker_id}") from None
    finally:
        response.close()


def run_scaling(registry, payload, timeout):
    try:
        reserved = ([registry.begin_single_worker_inference(timeout)] if payload.worker_count == 1
                    else registry.begin_two_worker_inference(WORKER_IDS, timeout))
    except UnknownWorkerError as exc:
        raise HTTPException(404, f"Worker not active: {exc}") from None
    except WorkerNotReadyError as exc:
        raise HTTPException(409, f"Worker not ready: {exc}") from None
    try:
        urls = [diagnostic_url(w) + "/scaling" for w, _ in reserved]
        if len(set(urls)) != len(urls):
            raise HTTPException(409, "Workers require independent diagnostic URLs")
        batches = [ScalingBatchRequest(**payload.model_dump(), partition=i) for i in range(payload.worker_count)]
        start = perf_counter()
        with ThreadPoolExecutor(max_workers=payload.worker_count) as pool:
            futures = [pool.submit(forward_scaling,w,url,batch)
                       for (w,_),url,batch in zip(reserved,urls,batches)]
            results = [f.result() for f in futures]
        elapsed = perf_counter()-start
        return ScalingResponse(**payload.model_dump(),workers=results,wall_time_ms=elapsed*1000,
            sum_worker_gpu_time_ms=sum(r.total_gpu_time_ms for r in results),
            effective_throughput_images_per_second=payload.total_samples/elapsed)
    finally:
        for worker,original in reserved:
            registry.finish_nn_test(worker.worker_id,original,worker)
