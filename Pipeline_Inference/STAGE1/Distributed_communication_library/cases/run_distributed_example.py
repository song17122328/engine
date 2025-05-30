import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def run_distributed_example(rank, world_size):
    """A simple distributed example where each process computes its rank and sends it to the next process."""
    # Initialize the process group
    # backend='gloo' is used for CPU-based communication,'gloo' is suitable for CPU and 'nccl' for GPU
    dist.init_process_group('nccl', rank=rank, world_size=world_size) # 'nccl' is used for GPU communication
    # dist.init_process_group('gloo', rank=rank, world_size=world_size)  # 'gloo' is used for CPU communication
    print(f'Rank {rank}/{world_size} : Process group initialized.')


def main():
    world_size = 2  # Number of processes
    os.environ['MASTER_ADDR'] = 'localhost'  # Address of the master node
    os.environ['MASTER_PORT'] = '29500'  # Use a random port for robustness

    mp.spawn(run_distributed_example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
