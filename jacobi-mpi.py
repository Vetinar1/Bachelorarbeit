import cupy.cuda.nccl as nccl
import matplotlib.pyplot as plt
from lib.kernels import *
import numpy as np
from mpi4py import MPI
import time

def mpi_solver(a_new, a_old, y_size, gpu_rank, ndevs, comm, blockspergrid, threadsperblock, cupy_kernels=False,
               max_iterations=np.inf, skipnorm=False):
    dist = 1
    it = 0
    up_idx = (gpu_rank - 1) % ndevs
    down_idx = (gpu_rank + 1) % ndevs
    while dist > 1e-5 and it < max_iterations:
        if it % 1000 == 0:
            print(it)

        it += 1
        if cupy_kernels:
            cp_jstep[blockspergrid, threadsperblock](a_old.ravel(), a_new.ravel(), a_old.shape[0], a_old.shape[1])
            # cp_jstep((blockspergrid,), (threadsperblock,), (a_old, a_new))
        else:
            nb_jstep[blockspergrid, threadsperblock](a_old, a_new)

        if not skipnorm:
            l2_norm    = cp.zeros(2, dtype=np.float32)
            l2_norm[0] = cp.linalg.norm(a_old) # TODO do not recalculate a_old
            l2_norm[1] = cp.linalg.norm(a_new)
            l2_norm[0] *= l2_norm[0]
            l2_norm[1] *= l2_norm[1]

            l2_norm_recv = cp.zeros_like(l2_norm)

            cp.cuda.get_current_stream().synchronize()
            comm.Allreduce(l2_norm, l2_norm_recv)

            l2_norm = cp.sqrt(l2_norm_recv)
            dist = cp.abs(l2_norm[0] - l2_norm[1])

        upper_row_send  = a_new[1]
        bottom_row_send = a_new[-2]
        upper_row_recv  = cp.zeros(y_size, dtype=np.float32)
        bottom_row_recv = cp.zeros(y_size, dtype=np.float32)

        cp.cuda.get_current_stream().synchronize()
        req1 = comm.Irecv(upper_row_recv, source=up_idx, tag=10)
        cp.cuda.get_current_stream().synchronize()
        comm.Send(bottom_row_send, dest=down_idx, tag=10)
        cp.cuda.get_current_stream().synchronize()
        req2 = comm.Irecv(bottom_row_recv, source=down_idx, tag=20)
        cp.cuda.get_current_stream().synchronize()
        comm.Send(upper_row_send, dest=up_idx, tag=20)

        req1.wait()
        req2.wait()

        a_new[0]  = upper_row_recv
        a_new[-1] = bottom_row_recv

        a_old = cp.array(a_new)

    print(f"Worker {gpu_rank} exiting loop")
    return a_old


def gpu_worker(ndevs, mp_rank, gpu_rank, x_size, y_size, comm, mode="cupy", max_iterations=None, skipnorm=False):
    if mp_rank != gpu_rank:
        print("WARNING: mp_rank ", mp_rank, " != gpu_rank ", gpu_rank)

    if max_iterations is None:
        max_iterations = np.inf

    print("max_iterations is", max_iterations)

    print(f"Worker {mp_rank} selecting CUDA device")
    cp.cuda.Device(gpu_rank).use() # TODO not sure if this is necessary
    print(f"Worker {mp_rank} on device", cp.cuda.get_device_id())

    print(f"Worker {mp_rank} creating cupy arrays")
    a_old = cp.zeros((x_size, y_size), dtype=np.float32)
    a_new = cp.zeros((x_size, y_size), dtype=np.float32)

    print(f"Worker {mp_rank} calculating grid")
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(a_new.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(a_new.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print(f"Worker {mp_rank} initalising arrays")
    nb_jinit[blockspergrid, threadsperblock](a_new, mp_rank * x_size, ndevs * x_size)
    nb_jinit[blockspergrid, threadsperblock](a_old, mp_rank * x_size, ndevs * x_size)

    t_start = time.time()
    if mode == "numba":
        a_old = mpi_solver(a_new, a_old, y_size, gpu_rank, ndevs, comm, blockspergrid, threadsperblock,
                           max_iterations=max_iterations, skipnorm=skipnorm)
    elif mode == "cupy":
        a_old = mpi_solver(a_new, a_old, y_size, gpu_rank, ndevs, comm, blockspergrid, threadsperblock,
                           max_iterations=max_iterations, skipnorm=skipnorm, cupy_kernels=True)
    else:
        raise RuntimeError("Invalid mode", mode)
    t_end = time.time()
    print(f"Iteration on worker {mp_rank}: {t_end - t_start:.2f}s")

    print(f"Worker {mp_rank} consolidating data...")
    a = cp.ones(x_size * ndevs * y_size, dtype=np.float32)
    # a_old = a_old[1:-1].flatten()
    a_old = a_old.flatten()

    comm.Allgather(a_old, a)
    a = a.reshape((x_size * ndevs, y_size))

    if mp_rank == 0:
        a_plot = cp.asnumpy(a)
        plt.imshow(a_plot)
        plt.colorbar()
        plt.savefig("test.png")

    print(f"Worker {mp_rank} done")


# Heavily inspired by pynccl example
if __name__ == "__main__":
    x_size = 16384
    y_size = 16384
    # x_size = 8192
    # y_size = 8192

    # set up MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    x_size_worker = math.ceil(x_size / size)
    y_size_worker = y_size

    max_iterations=100000
    skipnorm = True
    mode = "cupy"

    print(f"Started on {size} devices with x = {x_size} and y = {y_size}")
    gpu_worker(
        size,
        rank,
        rank,
        x_size_worker,
        y_size_worker,
        comm,
        mode,
        max_iterations,
        skipnorm
    )
