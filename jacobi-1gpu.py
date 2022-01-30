import cupy as cp
import cupyx as cpx
import numpy as np
import numba as nb
from numba import cuda
from numba import jit
import math
import matplotlib.pyplot as plt
import time


@cuda.jit
def jinit(a):
    tx, ty = cuda.grid(2)

    if (ty == 0) and tx < a.shape[0]:
        a[tx][ty] = -math.sin(2 * math.pi * tx / (a.shape[0] - 1))
    if (ty == a.shape[1] - 1) and tx < a.shape[0]:
        a[tx][ty] = math.sin(2 * math.pi * tx / (a.shape[0] - 1))

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

        # periodicity
        if tx == 1:
            b[a.shape[0] - 1][ty] = b[tx][ty]
        elif tx == a.shape[0] - 1:
            b[0][ty] = b[tx][ty]

    elif (ty == 0) and tx < a.shape[0]:
        b[tx][ty] = a[tx][ty]
    elif (ty == a.shape[1] - 1) and tx < a.shape[0]:
        b[tx][ty] = a[tx][ty]




def jacobi(x_size, y_size):
    a_old = cp.zeros((x_size, y_size), dtype=np.float32)
    a_new = cp.zeros((x_size, y_size), dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(a_new.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(a_new.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    jinit[blockspergrid, threadsperblock](a_new)
    jinit[blockspergrid, threadsperblock](a_old)

    t1 = time.time()

    i = 0
    dist = 1
    l2_old = None
    l2_new = None
    while dist > 1e-8 and i < 100:
        if i == 10:
            cp.cuda.profiler.start()
        cp.cuda.nvtx.RangePush(f"iteration {i}")
        i += 1
        cp.cuda.nvtx.RangePush(f"jstep")
        jstep[blockspergrid, threadsperblock](a_old, a_new)
        cp.cuda.nvtx.RangePop()

        if i % 10 == 0:
            cp.cuda.nvtx.RangePush(f"Norm calculation")
            l2_old = l2_new or np.inf
            l2_new = cp.linalg.norm(a_new)
            dist = abs(l2_old - l2_new)
            cp.cuda.nvtx.RangePop()

        a_old, a_new = a_new, a_old
        cp.cuda.nvtx.RangePop()


    cp.cuda.profiler.stop()
    print(i)

    a_plot = cp.asnumpy(a_old)
    print(round(time.time() - t1, 2), "s")

    # plt.imshow(a_plot)
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    jacobi(10000, 1000)
    # print(cpx.time.repeat(jacobi, (50, 100), n_repeat=100, max_duration=60))



