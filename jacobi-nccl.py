import cupy.cuda.nccl as nccl
import numpy as np
import numba as nb
from numba import cuda
import matplotlib.pyplot as plt
import multiprocessing as mp
from lib.kernels import *
from lib.solvers import *
import signal

def gpu_worker(ndevs, mp_rank, gpu_rank, q, x_size, y_size, mode):
    if mp_rank != gpu_rank:
        print("WARNING: mp_rank ", mp_rank, " != gpu_rank ", gpu_rank)

    # Generate unique id in thread 0 and share
    if mp_rank == 0:
        print("Worker 0 creating uid")
        uid = cp.cuda.nccl.get_unique_id()

        for i in range(ndevs-1):
            print("Worker 0 putting uid number", i)
            q.put(uid)
    else:
        print(f"Worker {mp_rank} receiving uid")
        uid = q.get()

    print(f"Worker {mp_rank} selecting CUDA device")
    cp.cuda.Device(gpu_rank).use() # TODO not sure if this is necessary
    print(f"Worker {mp_rank} on device", cp.cuda.get_device_id())

    print(f"Worker {mp_rank} creating NCCL communicator")
    nccl_comm = nccl.NcclCommunicator(ndevs, uid, gpu_rank) # TODO regular rank?


    print(f"Worker {mp_rank} creating cupy arrays")
    a_old = cp.zeros((x_size, y_size), dtype=np.float32)
    a_new = cp.zeros((x_size, y_size), dtype=np.float32)

    print(f"Worker {mp_rank} calculating grid")
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(a_new.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(a_new.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print(f"Worker {mp_rank} initalising arrays")
    nb_jinit[blockspergrid, threadsperblock](a_new, mp_rank * x_size, n_devs * x_size)
    nb_jinit[blockspergrid, threadsperblock](a_old, mp_rank * x_size, n_devs * x_size)

    if mode == "cp_nccl":
        a_old = nccl_solver(a_new, a_old, y_size, gpu_rank, ndevs, nccl_comm, blockspergrid, threadsperblock,
                            cupy_kernels=True)
    elif mode == "nb_nccl":
        a_old = nccl_solver(a_new, a_old, y_size, gpu_rank, ndevs, nccl_comm, blockspergrid, threadsperblock)
    elif mode == "nb_nccl_prio":
        a_old = nccl_priority_solver(a_new, a_old, y_size, gpu_rank, ndevs, nccl_comm, blockspergrid, threadsperblock)
    else:
        raise RuntimeWarning("invalid mode", mode)


    print(f"Worker {mp_rank} consolidating data...")
    a = cp.ones(( (x_size - 2) * ndevs, y_size), dtype=np.float32)
    a_old = a_old[1:-1].flatten()
    a_old = a_old.flatten() # TODO these numbers dont quite add up

    cp.cuda.nccl.groupStart()
    nccl_comm.allGather(a_old.data.ptr, a.data.ptr, x_size * y_size, nccl.NCCL_FLOAT32, cp.cuda.Stream.null.ptr)
    cp.cuda.nccl.groupEnd()

    if mp_rank == 0:
        a_plot = cp.asnumpy(a)
        plt.imshow(a_plot)
        plt.colorbar()
        plt.savefig("test.png")

    nccl_comm.destroy()

    print(f"Worker {mp_rank} done")


def launch_jacobi_workers(ndevs, x_size, y_size, mode):
    gpu_ranks = [i for i in range(ndevs)]
    q = mp.SimpleQueue()
    procs = []

    # Roughly
    x_size_worker = math.ceil(x_size / ndevs)
    y_size_worker = y_size
    print("x_size_worker =", x_size_worker)
    print("y_size_worker =", y_size_worker)

    print("Creating workers...")
    for i in range(ndevs):

        args = (ndevs, i, gpu_ranks[i], q, x_size_worker, y_size_worker, mode)
        worker = mp.Process(
            target=gpu_worker,
            args=args
        )
        procs.append(worker)

    def exit_handler(sig, frame):
        print("Ctrl C detected, cleaning up")
        for i, worker in enumerate(procs):
            print(i)
            worker.terminate()
        print("Exiting")
        exit()

    signal.signal(signal.SIGINT, exit_handler)

    for worker in procs:
        print("Starting worker", i)
        worker.start()

    for worker in procs:
        worker.join()


if __name__ == "__main__":
    n_devs = 4

    x_size = 400
    y_size = 100

    # valid: nb_nccl, cp_nccl, nb_nccl_prio
    mode = "nb_nccl"

    print(f"Started on {n_devs} devices with x = {x_size} and y = {y_size}")
    launch_jacobi_workers(
        ndevs=n_devs,
        x_size=x_size,
        y_size=y_size,
        mode=mode
    )
    # nb.cuda.profile_stop()
