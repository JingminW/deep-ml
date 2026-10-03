import numpy as np
import math

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    N, C, H, W = x.shape
    output_h = math.ceil((H - kernel_size) / stride) + 1
    output_w = math.ceil((W - kernel_size) / stride) + 1
    output = np.empty((N, C, output_h, output_w), dtype = x.dtype)

    out_i = 0
    i = 0

    while out_i < output_h:
        out_j = 0
        j = 0

        while out_j < output_w:
            patch = x[
                :,
                :,
                i:i + kernel_size,
                j:j + kernel_size
            ]

            kernel_max = np.max(
                patch,
                axis=(-2, -1)
            )

            output[:, :, out_i, out_j] = kernel_max

            j += stride
            out_j += 1

        i += stride
        out_i += 1

    return output
    
    