from .kernels import *
import cupy.cuda.nccl as nccl
import numpy as np
import numba as nb

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
                cupy_kernels=False):
    dist = 1
    it = 0
    while dist > 1e-5:
        it += 1
        if cupy_kernels:
            cp_jstep[blockspergrid, threadsperblock](a_old, a_new, a_old.shape[0], a_old.shape[1])
            # cp_jstep((blockspergrid,), (threadsperblock,), (a_old, a_new))
        else:
            nb_jstep[blockspergrid, threadsperblock](a_old, a_new)

        l2_norm    = cp.zeros(2, dtype=np.float32)
        l2_norm[0] = cp.linalg.norm(a_old) # TODO Avoid recalculating
        l2_norm[1] = cp.linalg.norm(a_new)
        l2_norm[0] *= l2_norm[0]
        l2_norm[1] *= l2_norm[1]
        cp.cuda.nccl.groupStart()
        nccl_comm.allReduce(l2_norm.data.ptr, l2_norm.data.ptr, 2, nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                            cp.cuda.Stream.null.ptr)
        cp.cuda.nccl.groupEnd()
        l2_norm = cp.sqrt(l2_norm)
        dist = cp.abs(l2_norm[0] - l2_norm[1])

        upper_row_send  = a_new[1]
        bottom_row_send = a_new[-2]
        upper_row_recv  = cp.zeros(y_size, dtype=np.float32)
        bottom_row_recv = cp.zeros(y_size, dtype=np.float32)

        up_idx = (gpu_rank - 1) % ndevs
        down_idx = (gpu_rank + 1) % ndevs

        cp.cuda.nccl.groupStart()
        nccl_comm.recv(upper_row_recv.data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   cp.cuda.Stream.null.ptr)
        nccl_comm.send(bottom_row_send.data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, cp.cuda.Stream.null.ptr)
        nccl_comm.recv(bottom_row_recv.data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, cp.cuda.Stream.null.ptr)
        nccl_comm.send(upper_row_send.data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   cp.cuda.Stream.null.ptr)
        cp.cuda.nccl.groupEnd()

        a_new[0]  = upper_row_recv
        a_new[-1] = bottom_row_recv

        a_old = cp.array(a_new)

    return a_old

def nccl_priority_solver(a_new, a_old, y_size, gpu_rank, ndevs, nccl_comm, blockspergrid, threadsperblock):
    comm_stream = cp.cuda.stream.Stream(priority=-5, non_blocking=True)
    comp_stream = cp.cuda.stream.Stream(priority=-1, non_blocking=True)

    dist = 1
    it = 0
    while dist > 1e-5:
        it += 1

        comm_stream.use()
        # calculation of first and last row requires neighboring rows
        nb_jstep[blockspergrid, threadsperblock](a_old[:3], a_new[:3])
        nb_jstep[blockspergrid, threadsperblock](a_old[-3:], a_new[-3:])

        comp_stream.use()
        # slices this way because the kernel excludes edges
        nb_jstep[blockspergrid, threadsperblock](a_old[1:-1], a_new[1:-1])

        comm_stream.synchronize()

        upper_row_send  = a_new[1]
        bottom_row_send = a_new[-2]
        # print(gpu_rank, it, "uppersend", upper_row_send)
        # print(gpu_rank, it, "lowersend", bottom_row_send)
        upper_row_recv  = cp.zeros(y_size, dtype=np.float32)
        bottom_row_recv = cp.zeros(y_size, dtype=np.float32)

        up_idx = (gpu_rank - 1) % ndevs
        down_idx = (gpu_rank + 1) % ndevs

        cp.cuda.nccl.groupStart()
        nccl_comm.recv(upper_row_recv.data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   comm_stream.ptr)
        nccl_comm.send(bottom_row_send.data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, comm_stream.ptr)
        nccl_comm.recv(bottom_row_recv.data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, comm_stream.ptr)
        nccl_comm.send(upper_row_send.data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   comm_stream.ptr)
        cp.cuda.nccl.groupEnd()

        comm_stream.synchronize()
        comp_stream.synchronize()

        a_new[0]  = upper_row_recv
        a_new[-1] = bottom_row_recv

        comm_stream.use()
        l2_norm    = cp.zeros(2, dtype=np.float32)
        l2_norm[0] = cp.linalg.norm(a_old)
        l2_norm[1] = cp.linalg.norm(a_new)
        l2_norm[0] *= l2_norm[0]
        l2_norm[1] *= l2_norm[1]
        cp.cuda.nccl.groupStart()
        nccl_comm.allReduce(l2_norm.data.ptr, l2_norm.data.ptr, 2, nccl.NCCL_FLOAT32, nccl.NCCL_SUM, comm_stream.ptr)
        cp.cuda.nccl.groupEnd()
        l2_norm = cp.sqrt(l2_norm)
        dist = cp.abs(l2_norm[0] - l2_norm[1])

        a_old = cp.array(a_new)

    return a_old

