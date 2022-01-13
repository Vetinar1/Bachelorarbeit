from numba import cuda
import cupy as cp
import math


@cuda.jit
def jinit(a, offset, fullsize):
    tx, ty = cuda.grid(2)

    if (ty == 0) and tx < a.shape[0]:
        a[tx][ty] = -math.sin(2 * math.pi * (tx + offset) / fullsize)
    if (ty == a.shape[1] - 1) and tx < a.shape[0]:
        a[tx][ty] =  math.sin(2 * math.pi * (tx + offset) / fullsize)

@cuda.jit
def jstep(a, b):
    """

    :param a:   old
    :param b:   new
    """
    tx, ty = cuda.grid(2)

    # Ignore first and last column because they are constant
    # Ignore first and last row because they come from neighbours / periodicity
    if 1 <= tx < a.shape[0] - 1 and 1 <= ty < a.shape[1] - 1:
        b[tx][ty] = 0.25 * (a[tx][ty + 1] + a[tx][ty - 1] + a[tx + 1][ty] + a[tx - 1][ty])

    elif (ty == 0) and tx < a.shape[0]:
        b[tx][ty] = a[tx][ty]
    elif (ty == a.shape[1] - 1) and tx < a.shape[0]:
        b[tx][ty] = a[tx][ty]


# from https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/nccl/jacobi_kernels.cu
cp_jinit = cp.RawKernel(
    """
    extern "C" __global__
    void initialize_boundaries(double * __restrict__ const a_new,
                               double * __restrict__ const a,
                               const int offset,
                               const int nx,
                               const int my_ny,
                               const int ny) {
        double pi = 3.1415926535;
        for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
            const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
            a[iy * nx + 0] = y0;
            a[iy * nx + (nx - 1)] = y0;
            a_new[iy * nx + 0] = y0;
            a_new[iy * nx + (nx - 1)] = y0;
        }
    }""",
    "initial_boundaries"
)


cp_jstep = cp.RawKernel(
    """
    template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
    __global__ void jacobi_kernel(double* __restrict__ const a_new,
                                  const double* __restrict__ const a,
                                  double* __restrict__ const l2_norm,
                                  const int iy_start,
                                  const int iy_end,
                                  const int nx,
                                  const bool calculate_norm) {
        int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
        int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
        real local_l2_norm = 0.0;

        if (iy < iy_end && ix < (nx - 1)) {
            const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                         a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;
            if (calculate_norm) {
                real residue = new_val - a[iy * nx + ix];
                local_l2_norm += residue * residue;
            }
        }
        if (calculate_norm) {
            atomicAdd(l2_norm, local_l2_norm);
        }
    }
    """,
    "jacobi_kernel"
)