from .kernels import *
import cupy.cuda.nccl as nccl
import numpy as np
import numba as nb
import cupy as cp

try:
    from mpi4py import MPI
except:
    MPI = None

# variants to try:
# - nccl
# - nccl with priority streams/communication overlap TODO
# - mpi4py TODO?
# - mpi4py (cuda aware)
# - (multiprocessing TODO)
# - numba TODO
# - (numba with priority streams) TODO

def nccl_solver(a_new, a_old, y_size, gpu_rank, ndevs, nccl_comm, blockspergrid, threadsperblock,
                cupy_kernels=False, max_iterations=np.inf, skipnorm=False):
    dist = 1
    it = 0
    up_idx = (gpu_rank - 1) % ndevs
    down_idx = (gpu_rank + 1) % ndevs
    l2_norm = cp.zeros(2, dtype=np.float32)
    l2_norm[1] = np.inf
    while dist > 1e-5 and it < max_iterations:
        if it == 10:
            cp.cuda.profiler.start()

        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} iteration {it}")
        it += 1
        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} jstep {it}")
        if cupy_kernels:
            cp_jstep[blockspergrid, threadsperblock](a_old.ravel(), a_new.ravel(), a_old.shape[0], a_old.shape[1])
        else:
            nb_jstep[blockspergrid, threadsperblock](a_old, a_new)
        cp.cuda.nvtx.RangePop()

        if not skipnorm:
            cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} norm {it}")

            l2_norm[0] = l2_norm[1]
            l2_norm[1] = cp.linalg.norm(a_new)
            l2_norm[0] *= l2_norm[0]
            l2_norm[1] *= l2_norm[1]
            cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} norm {it} comms")
            cp.cuda.nccl.groupStart()
            nccl_comm.allReduce(l2_norm.data.ptr, l2_norm.data.ptr, 2, nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                                cp.cuda.Stream.null.ptr)
            cp.cuda.nccl.groupEnd()
            cp.cuda.nvtx.RangePop()

            l2_norm = cp.sqrt(l2_norm)
            dist = cp.abs(l2_norm[0] - l2_norm[1])

            cp.cuda.nvtx.RangePop()

        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} comms {it}")
        cp.cuda.nccl.groupStart()
        # upper row recv
        nccl_comm.recv(a_new[0].data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   cp.cuda.Stream.null.ptr)
        # bottom row send
        nccl_comm.send(a_new[-2].data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, cp.cuda.Stream.null.ptr)
        # bottom row recv
        nccl_comm.recv(a_new[-1].data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, cp.cuda.Stream.null.ptr)
        # upper row send
        nccl_comm.send(a_new[1].data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   cp.cuda.Stream.null.ptr)
        cp.cuda.nccl.groupEnd()

        cp.cuda.nvtx.RangePop()

        a_old, a_new = a_new, a_old
        cp.cuda.nvtx.RangePop()

    cp.cuda.profiler.stop()
    print(f"Worker {gpu_rank} exiting solver after {it} iterations")
    return a_old

def nccl_priority_solver(a_new, a_old, y_size, gpu_rank, ndevs, nccl_comm, blockspergrid, threadsperblock,
                         cupy_kernels=False, max_iterations=np.inf, skipnorm=False):
    comm_stream = cp.cuda.stream.Stream(priority=-5, non_blocking=True)
    comp_stream = cp.cuda.stream.Stream(priority=-1, non_blocking=True)

    up_idx = (gpu_rank - 1) % ndevs
    down_idx = (gpu_rank + 1) % ndevs

    dist = 1
    it = 0
    l2_norm = cp.zeros(2, dtype=np.float32)
    l2_norm[1] = np.inf
    while dist > 1e-5 and it < max_iterations:
        if it == 10:
            cp.cuda.profiler.start()

        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} iteration {it}")
        it += 1

        comm_stream.use()
        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} edges {it}")
        # calculation of first and last row requires neighboring rows
        if cupy_kernels:
            cp_jstep[blockspergrid, threadsperblock](a_old[:3], a_new[:3])
            cp_jstep[blockspergrid, threadsperblock](a_old[-3:], a_new[-3:])
        else:
            nb_jstep[blockspergrid, threadsperblock](a_old[:3], a_new[:3])
            nb_jstep[blockspergrid, threadsperblock](a_old[-3:], a_new[-3:])
        cp.cuda.nvtx.RangePop()

        comp_stream.use()
        # slices this way because the kernel excludes edges
        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} body {it}")
        if cupy_kernels:
            cp_jstep[blockspergrid, threadsperblock](a_old[1:-1], a_new[1:-1])
        else:
            nb_jstep[blockspergrid, threadsperblock](a_old[1:-1], a_new[1:-1])
        cp.cuda.nvtx.RangePop()

        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} comm sync")
        comm_stream.synchronize()
        cp.cuda.nvtx.RangePop()

        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} comm prep")
        upper_row_recv  = cp.zeros(y_size, dtype=np.float32)
        bottom_row_recv = cp.zeros(y_size, dtype=np.float32)
        cp.cuda.nvtx.RangePop()

        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} comm exec")
        cp.cuda.nccl.groupStart()
        nccl_comm.recv(upper_row_recv.data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   comm_stream.ptr)
        # bottom row send
        nccl_comm.send(a_new[-2].data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, comm_stream.ptr)
        nccl_comm.recv(bottom_row_recv.data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, comm_stream.ptr)
        # upper row send
        nccl_comm.send(a_new[1].data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   comm_stream.ptr)
        cp.cuda.nccl.groupEnd()
        cp.cuda.nvtx.RangePop()

        cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} sync")
        comm_stream.synchronize()
        comp_stream.synchronize()
        cp.cuda.nvtx.RangePop()

        a_new[0]  = upper_row_recv
        a_new[-1] = bottom_row_recv

        if not skipnorm:
            cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} norm calc")
            comm_stream.use()
            l2_norm    = cp.zeros(2, dtype=np.float32)
            l2_norm[0] = l2_norm[1]
            l2_norm[1] = cp.linalg.norm(a_new)
            l2_norm[0] *= l2_norm[0]
            l2_norm[1] *= l2_norm[1]
            cp.cuda.nvtx.RangePush(f"GPU {gpu_rank} norm comm")
            cp.cuda.nccl.groupStart()
            nccl_comm.allReduce(l2_norm.data.ptr, l2_norm.data.ptr, 2, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, comm_stream.ptr)
            cp.cuda.nccl.groupEnd()
            cp.cuda.nvtx.RangePop()
            l2_norm = cp.sqrt(l2_norm)
            dist = cp.abs(l2_norm[0] - l2_norm[1])
            cp.cuda.nvtx.RangePop()

        a_old = cp.array(a_new)
        cp.cuda.nvtx.RangePop()

    cp.cuda.profiler.stop()
    print(f"Worker {gpu_rank} exiting solver after {it} iterations")
    return a_old

