import torch
import time

print("=" * 50)
print("ColabCluster - T4 GPU Exercise")
print("=" * 50)

# 1. GPU detection
print("\n[1] GPU INFORMATION")
print("CUDA available :", torch.cuda.is_available())
print("GPU            :", torch.cuda.get_device_name(0))
print("CUDA version   :", torch.version.cuda)

# 2. Allocate matrices on GPU
device = torch.device("cuda")

size = 4096

print("\n[2] ALLOCATING GPU MATRICES")
print(f"Matrix size    : {size} x {size}")

A = torch.randn(size, size, device=device)
B = torch.randn(size, size, device=device)

print("GPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

# 3. Warm-up
print("\n[3] GPU WARM-UP")

for _ in range(10):
    C = torch.matmul(A, B)

torch.cuda.synchronize()

# 4. Benchmark
print("\n[4] MATRIX MULTIPLICATION BENCHMARK")

iterations = 50

start = time.perf_counter()

for _ in range(iterations):
    C = torch.matmul(A, B)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print("Iterations       :", iterations)
print("Total time       :", round(elapsed, 3), "seconds")
print("Average per op   :", round(elapsed / iterations * 1000, 2), "ms")

# 5. GPU utilization / memory
print("\n[5] GPU MEMORY")

print("Allocated:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

print("Reserved:",
      round(torch.cuda.memory_reserved() / 1024**3, 2), "GB")

# 6. Verify result
print("\n[6] COMPUTATION CHECK")

print("Result shape:", C.shape)
print("Result device:", C.device)
print("Result mean:", C.mean().item())

# 7. Cleanup
del A, B, C
torch.cuda.empty_cache()

print("\n[7] CLEANUP")
print("GPU memory after cleanup:",
      round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

print("\n" + "=" * 50)
print("T4 GPU EXERCISE COMPLETE")
print("=" * 50)