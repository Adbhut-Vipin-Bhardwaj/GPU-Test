import torch
import time

def benchmark_matmul(device, size=4096, iterations=100):
    print(f"Benchmarking on {device}...")
    try:
        # Create random matrices
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warm-up to ensure accurate timing
        for _ in range(10):
            _ = torch.matmul(a, b)

        # Synchronization is important for asynchronous devices like xpu
        if device.type == 'xpu':
            torch.xpu.synchronize()

        start_time = time.time()

        # The actual benchmark loop
        for _ in range(iterations):
            _ = torch.matmul(a, b)

        # Wait for all device operations to finish before taking the end time
        if device.type == 'xpu':
            torch.xpu.synchronize()

        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        print(f"Average time per matmul ({size}x{size}) on {device}: {avg_time*1000:.2f} ms")
        return avg_time

    except Exception as e:
        print(f"Error benchmarking on {device}: {e}")
        return None

if __name__ == "__main__":
    matrix_size = 4096
    num_iterations = 50

    print(f"--- Speed Comparison: CPU vs XPU ---")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Iterations: {num_iterations}\n")

    # CPU Benchmark
    cpu_device = torch.device("cpu")
    cpu_time = benchmark_matmul(cpu_device, size=matrix_size, iterations=num_iterations)
    print()

    # XPU Benchmark
    if torch.xpu.is_available():
        xpu_device = torch.device("xpu")
        xpu_time = benchmark_matmul(xpu_device, size=matrix_size, iterations=num_iterations)
        print()

        if cpu_time and xpu_time:
            speedup = cpu_time / xpu_time
            print(f"Summary: XPU is {speedup:.2f}x faster than CPU")
    else:
        print("Intel XPU is not available. Cannot perform comparison.")
