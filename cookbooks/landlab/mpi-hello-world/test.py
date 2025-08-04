from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sum = comm.allreduce(rank, op=MPI.SUM)

if rank == 0:
    print(f"Hello World! I am process {rank} of {size}, sum = {sum}")

