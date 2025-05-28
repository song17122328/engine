"""
Once tensors are on the GPU, most PyTorch operations work seamlessly, leveraging the GPU's parallel processing capabilities.
"""
import torch

if torch.cuda.is_available():
    device=torch.device("cuda:0")  # Use the first GPU
    print(f"Performing operations on {device}")

    # Create two tensors on GPU
    tensor_a=torch.tensor([[1.0,2.0],[3.0,4.0]],device=device)
    tensor_b=torch.tensor([[5.0,6.0],[7.0,8.0]],device=device)

    print(f"\nTensor A on GPU:\n{tensor_a}")
    print(f"Tensor B on GPU:\n{tensor_b}")

    # Element-wise addition
    sum_tensor = tensor_a+tensor_b
    print(f"\nSum (A+B): \n{sum_tensor}\nDevice:{sum_tensor.device}")

    # Matrix multiplication
    matmul_tensor = torch.matmul(tensor_a, tensor_b)
    print(f"\nMatrix Multiplication (A @ B): \n{matmul_tensor}\nDevice:{matmul_tensor.device}")

    #Scalar multiplication
    scaled_tensor=tensor_a * 2.5
    print(f"\nScaled Tensor A (A * 2.5): \n{scaled_tensor}\nDevice:{scaled_tensor.device}")

    # Move result back to CPU if needed
    cpu_result = matmul_tensor.to("cpu")
    print(f"\nMatrix Multiplication result moved back to CPU:\n{cpu_result}\nDevice:{cpu_result.device}")
else:
    print("CUDA is not available. Cannot demonstrate GPU tensor operations.")