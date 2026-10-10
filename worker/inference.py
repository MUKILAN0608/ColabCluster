"""Fixed two-partition synthetic inference using the existing SmallCNN builder."""
import importlib
import time
from common.schemas import InferenceBatchResponse, SingleInferenceResponse
from common.scaling import ScalingBatchRequest, ScalingBatchResponse, partition_samples
from worker.cnn_test import build_small_cnn, CnnTestError


def run_inference(worker_id: str, partition: int, *, single: bool = False, scaling: ScalingBatchRequest | None = None) -> InferenceBatchResponse | SingleInferenceResponse:
    """Run ten warmups and one timed float32 forward, returning real GPU metrics."""
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
            if scaling is not None:
                batch_size = scaling.assigned_samples
                offset = sum(partition_samples(
                    scaling.total_samples, scaling.worker_count)[:scaling.partition])
                chunks = []
                # Global 32-sample blocks give identical inputs for one/two workers.
                for block in range(offset // 32, (offset + batch_size) // 32):
                    generator = torch.Generator(device="cuda:0").manual_seed(1000 + block)
                    chunks.append(torch.randn(32, 3, 32, 32, device="cuda:0",
                                              dtype=torch.float32, generator=generator))
                inputs = torch.cat(chunks, dim=0)
                del chunks
            else:
                generator = torch.Generator(device="cuda:0").manual_seed(1000 if single else 1000 + partition)
                inputs = torch.randn(32, 3, 32, 32, device="cuda:0", dtype=torch.float32,
                                     generator=generator)
                if single:
                    # Reconstruct exactly the two existing synthetic partitions locally.
                    generator.manual_seed(1001)
                    second = torch.randn(32, 3, 32, 32, device="cuda:0", dtype=torch.float32,
                                         generator=generator)
                    inputs = torch.cat((inputs, second), dim=0)
                    del second
                batch_size = 64 if single else 32
            for _ in range(10):
                output = model(inputs)
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
            start = time.perf_counter()
            output = model(inputs)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - start
            if elapsed <= 0:
                raise CnnTestError("CUDA timing did not produce a positive duration")
            response_type = ScalingBatchResponse if scaling is not None else SingleInferenceResponse if single else InferenceBatchResponse
            return response_type(
                **({"partition": scaling.partition} if scaling is not None else {} if single else {"partition": partition}), cuda_available=True,
                worker_id=worker_id, gpu=torch.cuda.get_device_name(0),
                cuda_version=str(torch.version.cuda), torch_version=str(torch.__version__),
                device=str(output.device), model="SmallCNN", input_shape=tuple(inputs.shape),
                batch_size=batch_size, forward_passes=1, total_gpu_time_ms=elapsed * 1000,
                average_inference_ms=elapsed * 1000,
                throughput_images_per_second=batch_size / elapsed,
                peak_memory_mb=torch.cuda.max_memory_allocated(0) / 1024**2,
                output_shape=tuple(output.shape), status="passed",
            )
    except CnnTestError:
        raise
    except Exception as exc:
        raise CnnTestError(f"SmallCNN CUDA diagnostic failed: {exc}") from exc
    finally:
        del model, inputs, output, generator
