from numba import cuda
import cupy as cp
import math
from cupyx import jit as cpjit


@cuda.jit
def nb_jinit(a, offset, fullsize):
    tx, ty = cuda.grid(2)

    if (ty == 0) and tx < a.shape[0]:
        a[tx][ty] = -math.sin(2 * math.pi * (tx + offset) / fullsize)
    if (ty == a.shape[1] - 1) and tx < a.shape[0]:
        a[tx][ty] =  math.sin(2 * math.pi * (tx + offset) / fullsize)

@cuda.jit
def nb_jstep(a, b):
    """

    :param a:   old
    :param b:   new
    """
    tx, ty = cuda.grid(2)

    # Ignore first and last column because they are constant
    # Ignore first and last row because they come from neighbours / periodicity
    # TODO Why no - 1??
    if 1 <= tx < a.shape[0] and 1 <= ty < a.shape[1] - 1:
        b[tx][ty] = 0.25 * (a[tx][ty + 1] + a[tx][ty - 1] + a[tx + 1][ty] + a[tx - 1][ty])

    elif (ty == 0) and tx < a.shape[0]:
        b[tx][ty] = a[tx][ty]
    elif (ty == a.shape[1] - 1) and tx < a.shape[0]:
        b[tx][ty] = a[tx][ty]


# @cpjit.rawkernel()
# def cp_jinit(a, offset, fullsize):
#     tx, ty = cpjit.grid(2)
#
#     if (ty == 0) and tx < a.shape[0]:
#         a[tx][ty] = -math.sin(2 * math.pi * (tx + offset) / fullsize)
#     if (ty == a.shape[1] - 1) and tx < a.shape[0]:
#         a[tx][ty] =  math.sin(2 * math.pi * (tx + offset) / fullsize)

@cpjit.rawkernel()
def cp_jstep(a, b, size_x, size_y):
    """

    :param a:   old
    :param b:   new
    """
    tx, ty = cpjit.grid(2)

    index = tx * size_y + ty

    # Ignore first and last column because they are constant
    # Ignore first and last row because they come from neighbours / periodicity
    if 1 <= tx and tx < size_x - 1 and 1 <= ty and ty < size_y - 1:
        # b[tx][ty] = 0.25 * (a[tx][ty + 1] + a[tx][ty - 1] + a[tx + 1][ty] + a[tx - 1][ty])
        b[index] = 0.25 * (a[index + 1] + a[index - 1] + a[index + size_y] + a[index - size_y])

    elif (ty == 0) and tx < size_x:
        # b[tx][ty] = a[tx][ty]
        b[index] = a[index]
    elif (ty == size_y - 1) and tx < size_x:
        # b[tx][ty] = a[tx][ty]
        b[index] = a[index]
