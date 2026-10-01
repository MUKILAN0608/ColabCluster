"""Run from Windows: python testing/gpu_test.py --worker-id COLAB-01."""

import argparse
import os
import sys
from urllib.parse import quote

import requests


def main() -> int:
    """Request remote computation without importing PyTorch."""
    parser = argparse.ArgumentParser(description="ColabCluster Remote GPU Test")
    parser.add_argument("--worker-id", default=os.getenv("COLABCLUSTER_WORKER_ID"))
    parser.add_argument("--controller-url", default=os.getenv("COLABCLUSTER_CONTROLLER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not 256 <= args.matrix_size <= 8192 or not 1 <= args.iterations <= 100:
        parser.error("matrix-size must be 256..8192 and iterations 1..100")
    base = args.controller_url.rstrip("/")
    try:
        worker_id = args.worker_id
        if not worker_id:
            response = requests.get(base + "/workers", timeout=10)
            response.raise_for_status()
            candidates = [w for w in response.json()["workers"]
                          if w.get("cuda_available") and (w.get("metadata") or {}).get("diagnostic_url")]
            if len(candidates) != 1:
                raise ValueError("Specify --worker-id; exactly one diagnostic-enabled GPU worker was not found")
            worker_id = candidates[0]["worker_id"]
        response = requests.post(
            base + "/workers/" + quote(worker_id, safe="") + "/gpu-test",
            json={"matrix_size": args.matrix_size, "iterations": args.iterations}, timeout=(5, 135),
        )
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}: {response.text[:500]}")
        result = response.json()
        if (result["worker_id"] != worker_id or not result["cuda_available"]
                or result["result_shape"] != [args.matrix_size, args.matrix_size]):
            raise ValueError("Unexpected GPU benchmark result")
        print("=" * 50 + "\nColabCluster Remote GPU Test\n" + "=" * 50)
        for label, key in [("Worker", "worker_id"), ("GPU", "gpu"), ("CUDA", "cuda_version"),
                           ("PyTorch", "torch_version"), ("Matrix size", "matrix_size"),
                           ("Iterations", "iterations"), ("Total time (s)", "total_time_seconds"),
                           ("Average time (ms)", "average_time_ms"),
                           ("GPU memory (GiB)", "gpu_memory_allocated_gb"), ("Result shape", "result_shape")]:
            print(f"{label:20}: {result[key]}")
        print("GPU TEST PASSED\n" + "=" * 50)
        return 0
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"GPU TEST FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
