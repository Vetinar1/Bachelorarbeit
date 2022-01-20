from .kernels import *
import cupy.cuda.nccl as nccl
import numpy as np
import numba as nb

# variants to try:
# - nccl
# - nccl with priority streams/communication overlap TODO
# - mpi4py TODO
# - multiprocessing TODO
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
    comm_stream = cp.cuda.stream.Stream(priority=-5)
    comp_stream = cp.cuda.stream.Stream(priority=-1)

    dist = 1
    it = 0
    while dist > 1e-5:
        it += 1

        comm_stream.use()
        cp_jstep[blockspergrid, threadsperblock](a_old[1], a_new[1])
        cp_jstep[blockspergrid, threadsperblock](a_old[-2], a_new[-2])

        upper_row_send  = a_new[1]
        bottom_row_send = a_new[-2]
        upper_row_recv  = cp.zeros(y_size, dtype=np.float32)
        bottom_row_recv = cp.zeros(y_size, dtype=np.float32)

        comp_stream.use()
        cp_jstep[blockspergrid, threadsperblock](a_old[1:-2], a_new[1:-2])

        up_idx = (gpu_rank - 1) % ndevs
        down_idx = (gpu_rank + 1) % ndevs

        cp.cuda.nccl.groupStart()
        nccl_comm.recv(upper_row_recv.data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   comm_stream.ptr)
        nccl_comm.send(bottom_row_send.data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, comm_stream.ptr)
        nccl_comm.recv(bottom_row_recv.data.ptr, y_size, nccl.NCCL_FLOAT32, down_idx, comm_stream.ptr)
        nccl_comm.send(upper_row_send.data.ptr,  y_size, nccl.NCCL_FLOAT32, up_idx,   comm_stream.ptr)
        cp.cuda.nccl.groupEnd()

        a_new[0]  = upper_row_recv
        a_new[-1] = bottom_row_recv

        l2_norm    = cp.zeros(2, dtype=np.float32)
        l2_norm[0] = cp.linalg.norm(a_old)
        l2_norm[1] = cp.linalg.norm(a_new)
        l2_norm[0] *= l2_norm[0]
        l2_norm[1] *= l2_norm[1]
        cp.cuda.nccl.groupStart()
        nccl_comm.allReduce(l2_norm.data.ptr, l2_norm.data.ptr, 2, nccl.NCCL_FLOAT32, nccl.NCCL_SUM,
                            cp.cuda.Stream.null.ptr)
        cp.cuda.nccl.groupEnd()
        l2_norm = cp.sqrt(l2_norm)
        dist = cp.abs(l2_norm[0] - l2_norm[1])

        a_old = cp.array(a_new)

    return a_old

