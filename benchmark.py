import torch
import time
import argparse
import numpy as np

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

try:
    import intel_extension_for_pytorch as ipex
    from torch import xpu
except ImportError:
    pass
from torch import mps, cuda


parser = argparse.ArgumentParser(description="Measure FLOPs and BW.")
parser.add_argument(
    "--device", type=str, default="cpu", help="One of cpu | cuda | mps | xpu"
)
parser.add_argument(
    "--num_trails", type=int, default=100, help="Number of trails to get average."
)
parser.add_argument(
    "--dtype",
    type=str,
    default="float",
    help="One of float32|float64|float16|bfloat16|int8|int16|int32|bool",
)
args = parser.parse_args()

dtype = getattr(torch, args.dtype)
device = torch.device(args.device)
num_trails = args.num_trails
eps = 1e-6


def flops_benchmark(device):
    print(f'Device "{device}" {dtype} matmul compute benchmark: ')

    tflops_log = []
    test_range = 2 ** np.arange(11, 13 + eps, 0.25)  # 64~4096
    for n in test_range:
        n = int(n)
        a = 10 * torch.rand(n, n, device=device)
        a = a.to(dtype)

        # warmup
        for _ in range(3):
            torch.matmul(a, a)

        t = 0
        for _ in range(num_trails):
            synchronize(device)
            now = time.perf_counter()
            torch.matmul(a, a)
            synchronize(device)
            t += time.perf_counter() - now

        t /= num_trails

        tflops = 2 * n**3 / t / 1e12
        tflops_log.append(tflops)

        # print(n, t, tflops, sep=", ")
        print(f"Tensor size {n:,}, Elapsed time {t:.4e}s, {tflops:.4f} TOPs.")
    print(
        f'Device "{device}" yields {np.mean(tflops_log).item():.4f} TOPs on {dtype} (min: {np.min(tflops_log).item():.4f} TOPs, max: {np.max(tflops_log).item():.4f} TOPs)\n'
    )


def synchronize(device):
    if device.type == "cuda":
        cuda.synchronize()
    elif device.type == "mps":
        mps.synchronize()
    elif device.type == "xpu":
        xpu.synchronize()
    elif device.type == "cpu":
        pass


def memory_bandwidth_benchmark(device):
    print(f'Device "{device}" memory bandwidth benchmark: ')

    bandwidth_log = []
    test_range = 2 ** (np.arange(24, 30 + eps, 1))
    for size in test_range:
        elapsed_time = 0
        for _ in range(num_trails):
            size = int(size)

            # Create random tensors
            a = torch.rand(size, device=device)
            b = torch.rand(size, device=device)

            # Warm-up to ensure CUDA kernel is initialized if using GPU
            synchronize(device)
            a.copy_(b)
            synchronize(device)

            # Record the start time
            start_time = time.perf_counter()

            # Perform the copy operation
            a.copy_(b)

            # Synchronize if using CUDA to make sure operation is finished
            synchronize(device)

            # Record the end time
            end_time = time.perf_counter()

            # Compute elapsed time
            elapsed_time += end_time - start_time

        elapsed_time = elapsed_time / num_trails
        # Calculate Bandwidth in GB/s
        bytes_copied = a.nelement() * a.element_size()  # bytes
        bandwidth = 2 * bytes_copied / elapsed_time / 1e9  # GB/s
        bandwidth_log.append(bandwidth)
        print(
            f"Copy size {bytes_copied/1e9:.4f} GB, Elapsed time {elapsed_time:.4e}s, Bandwidth {bandwidth:.4f} GB/s"
        )
    print(
        f'Device "{device}" yields {np.mean(bandwidth_log).item():.4f} GB/s (min: {np.min(bandwidth_log).item():.4f} GB/s, max: {np.max(bandwidth_log).item():.4f} GB/s)\n'
    )
    return bandwidth


if __name__ == "__main__":
    flops_benchmark(device)
    memory_bandwidth_benchmark(device)
