import numpy as np
import time
from numba import cuda

N = 2**20
ITERATIONS = 1000
THREADS_PER_BLOCK = 256

@cuda.jit
def uniform_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        value = data[idx]
        for _ in range(ITERATIONS):
            value = value * 1.0001 + 0.0001
        data[idx] = value
