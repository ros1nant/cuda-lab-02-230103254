import numpy as np
from numba import cuda

@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)

    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            gx = (
                -1.0 * d_in[row - 1, col - 1]
                +1.0 * d_in[row - 1, col + 1]
                -2.0 * d_in[row, col - 1]
                +2.0 * d_in[row, col + 1]
                -1.0 * d_in[row + 1, col - 1]
                +1.0 * d_in[row + 1, col + 1]
            )

            d_out[row, col] = gx
        else:
            d_out[row, col] = 0.0

def run_sobel(h_img):
    h_img = np.asarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(d_in)

    threads_2d = (16, 16)

    blocks_2d = (
        (cols + threads_2d[0] - 1) // threads_2d[0],
        (rows + threads_2d[1] - 1) // threads_2d[1]
    )

    sobel_x_kernel[blocks_2d, threads_2d](
        d_in, d_out, rows, cols
    )

    cuda.synchronize()

    return d_out.copy_to_host()

if __name__ == "__main__":
    h_img = np.ones((2048, 2048), dtype=np.float32)
    result = run_sobel(h_img)

    assert np.allclose(result, 0.0)

    print("TASK 4 PASSED")
    print("Input shape:", h_img.shape)
    print("Output shape:", result.shape)
    print("Maximum absolute gradient:", np.max(np.abs(result)))
    print("Border pixels zero:", np.all(result[0, :] == 0))
