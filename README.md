# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103254  
**Allocated GPU Node:** NVIDIA Tesla T4  
**CUDA Compute Capability:** 7.5  
**CUDA Version:** 13.0  
**Official Verification Token:** E6A84AB5DAD03ABFD77D

## 1. Project Overview

This laboratory covers advanced CUDA programming using Python and Numba.

The laboratory consists of four tasks:

1. Warp Divergence Microbenchmark
2. 1D Boundary Stencil and Halo Protection
3. Grid-Stride Vector Scaling
4. 2D Sobel-X Convolution

## 2. Hardware Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 15 GB |
| Compute Capability | 7.5 |
| CUDA Version | 13.0 |
| NVIDIA Driver | 580.82.07 |
| NumPy | 2.1.3 |
| Numba | 0.61.2 |
| Environment | Google Colab |

CUDA availability was confirmed using Numba.

## 3. Task 1: Warp Divergence

### Objective

Investigate how conditional branching affects GPU performance.

A CUDA warp consists of 32 threads.

### Kernel A: Uniform Path

All threads execute the same arithmetic instructions for 1,000 iterations.

### Kernel B: Full Divergence

Even-indexed threads execute multiply-accumulate operations, while odd-indexed threads execute subtract-divide operations.

### Kernel C: Warp-Aligned Branching

Threads branch according to their warp ID. All threads within the same warp execute the same branch.

### Benchmark Configuration

- Input elements: 1,048,576
- Data type: float32
- Threads per block: 256
- Iterations: 1,000
- Warm-up launches: 1
- Benchmark trials: 10
- Timing: time.perf_counter()
- Synchronization: cuda.synchronize()

### Benchmark Results

| Kernel | Average Time (ms) |
|---|---|
| Uniform Path | Pending |
| Full Divergence | Pending |
| Warp-Aligned | Pending |

### Analysis

Warp divergence occurs when threads within the same warp follow different execution branches.

Warp-aligned branching reduces intra-warp divergence because all 32 threads follow the same execution path.

Actual execution-time comparisons require completed benchmark measurements.

## 4. Task 2: 1D Boundary Stencil

### Objective

Implement a three-point smoothing filter with boundary clamping.

### Formula

output[i] = 0.25 * input[i-1] + 0.5 * input[i] + 0.25 * input[i+1]

### Implementation

- CUDA indexing: cuda.grid(1)
- Input size: 10,007
- Threads per block: 256
- Data type: float32
- CPU validation: NumPy edge padding
- Tolerance: 1e-4

Boundary clamping replicates the first and last array values when neighboring indices fall outside the valid range.

### Results

TASK 2 PASSED: MAX DELTA = 0.0000000596

**Status: PASS**

### Explanation

The GPU output was compared with the NumPy CPU reference.

The maximum numerical difference was 0.0000000596, which satisfied the required tolerance.

## 5. Task 3: Grid-Stride Vector Scaling

### Objective

Process a large input vector using a fixed CUDA launch configuration.

### Implementation

Each thread calculates:

- Starting index using cuda.grid(1)
- Grid stride using cuda.gridsize(1)

Each thread processes elements separated by the total grid stride.

### Configuration

| Parameter | Value |
|---|---|
| Input Elements | 16,777,216 |
| Threads per Block | 256 |
| Blocks per Grid | 64 |
| Total Threads | 16,384 |
| Scaling Factor | 4.25 |

### Results

TASK 3 PASSED

Total elements: 16777216

Threads launched: 16384

Scaling factor: 4.25

All values correct: True

**Status: PASS**

### Explanation

Grid-stride loops allow a fixed number of threads to process a much larger number of elements.

All 16,777,216 elements were successfully processed and validated.

## 6. Task 4: 2D Sobel-X Convolution

### Objective

Implement a two-dimensional CUDA kernel for image gradient calculation.

### Sobel-X Matrix

| -1 | 0 | +1 |
|---|---|---|
| -2 | 0 | +2 |
| -1 | 0 | +1 |

### Configuration

- Image dimensions: 2048 x 2048
- Data type: float32
- Block dimensions: (16, 16)
- Indexing: cuda.grid(2)
- Border values: 0.0

### Results

TASK 4 PASSED

Input shape: (2048, 2048)

Output shape: (2048, 2048)

Maximum absolute gradient: 0.0

Border pixels zero: True

**Status: PASS**

### Explanation

A uniform input image was used for validation.

Since the image contains no intensity changes, the Sobel-X output is expected to be zero.

The GPU implementation produced zero gradients and correctly handled the border pixels.

## 7. Automated Verification

The official verification script produced the following results:

- [PASS] Task 2 (1D Stencil & Clamping)
- [PASS] Task 3 (Grid-Stride Scaling)
- [PASS] Task 4 (2D Sobel Horizontal)

**VERIFICATION SUCCESSFUL**

**Official Submission Token: E6A84AB5DAD03ABFD77D**

The verification token was generated using SHA-256 based on the student ID, CUDA device information, and selected GPU computation outputs.

## 8. Project Structure

- README.md
- task1_divergence.py
- task2_stencil_1d.py
- task3_grid_stride.py
- task4_sobel_2d.py
- verify_submission.py

## 9. How to Run

### Requirements

Python, NumPy, Numba with CUDA support, and an NVIDIA CUDA-compatible GPU.

### Run Individual Tasks

    python task1_divergence.py
    python task2_stencil_1d.py
    python task3_grid_stride.py
    python task4_sobel_2d.py

### Run Automated Verification

    python verify_submission.py

Enter Student ID: 230103254

## 10. Conclusion

This laboratory demonstrates CUDA programming concepts including warp divergence, boundary clamping, grid-stride loops, and two-dimensional convolution.

Task 2 successfully matched the CPU reference.

Task 3 processed 16,777,216 elements using 16,384 CUDA threads.

Task 4 produced the expected zero gradient for a uniform image.

The official automated verification passed Tasks 2, 3, and 4.

Task 1 benchmark measurements are still pending.
