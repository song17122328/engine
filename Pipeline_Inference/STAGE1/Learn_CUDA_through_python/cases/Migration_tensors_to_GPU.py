"""Master the migration of tensors to GPU (`tensor.to('cuda:0')`): PyTorch tensors can be easily moved between
CPU and GPU memory. `cuda:0` refers to the first GPU available. If you have multiple GPUs, you can use `cuda:1`,
`cuda:2`, etc."""

import torch

# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first GPU
    print(f"CUDA is available! Using device :{device}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")

# Create tensor on CPU
cpu_tensor = torch.rand(3, 3)
print(f"\n CPU Tensor: \n {cpu_tensor}\n Device: {cpu_tensor.device}")

# Move tensor to GPU
if device.type == 'cuda':
    gpu_tensor = cpu_tensor.to(device)
    print(f"\n GPU Tensor: \n {gpu_tensor}\n Device: {gpu_tensor.device}")

    # It can also create a tensor directly on the GPU
    direct_gpu_tensor = torch.zeros(2, 2, device=device)
    print(f"\n Directly created GPU Tensor: \n {direct_gpu_tensor}\n Device: {direct_gpu_tensor.device}")
