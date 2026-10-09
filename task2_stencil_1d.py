import numpy as np
from numba import cuda

@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    if idx < N:
        left = max(idx - 1, 0)
        right = min(idx + 1, N - 1)

        d_out[idx] = (
            0.25 * d_in[left]
            + 0.5 * d_in[idx]
            + 0.25 * d_in[right]
        )

def run_stencil(h_in):
    h_in = np.asarray(h_in, dtype=np.float32)
    N = len(h_in)

    if N == 0:
        return np.empty(0, dtype=np.float32)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    stencil_1d[blocks_per_grid, threads_per_block](d_in, d_out, N)
    cuda.synchronize()

    return d_out.copy_to_host()

def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode="edge")
    return (
        0.25 * padded[:-2]
        + 0.5 * padded[1:-1]
        + 0.25 * padded[2:]
    )

if __name__ == "__main__":
    N = 10007
    h_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)

    h_gpu = run_stencil(h_in)
    h_cpu = cpu_stencil(h_in)

    max_delta = np.max(np.abs(h_gpu - h_cpu))

    assert np.allclose(h_gpu, h_cpu, atol=1e-4)
    print(f"TASK 2 PASSED: MAX DELTA = {max_delta:.10f}")
