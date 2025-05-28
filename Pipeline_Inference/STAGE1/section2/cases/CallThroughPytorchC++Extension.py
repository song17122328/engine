"""**Call through PyTorch C++ extension or CuPy:** Calling custom CUDA kernels from Python can be done using
PyTorch's C++ extensions or libraries like CuPy. CuPy is generally simpler for raw kernel invocation."""
import cupy as cp
import numpy as np
import math

if cp.cuda.is_available():
    # Define the CUDA kernel as a string
    vector_add_kernel_code = r'''
    extern "C" __global__ void vector_add(const float* A, const float* B, float* C, int numElements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numElements) {
            C[idx] = A[idx] + B[idx];
        }
    }
    '''

    # Create a RawKernel object
    # The first argument is the CUDA code, the second is the kernel function name
    vector_add_kernel = cp.RawKernel(vector_add_kernel_code, 'vector_add')

    # Prepare host data
    num_elements = 1000000
    a_host = np.random.rand(num_elements).astype(np.float32)
    b_host = np.random.rand(num_elements).astype(np.float32)
    c_host = np.zeros(num_elements).astype(np.float32)

    # Transfer data to device
    a_gpu = cp.asarray(a_host)
    b_gpu = cp.asarray(b_host)
    c_gpu = cp.asarray(c_host)

    # Define kernel launch configuration
    threads_per_block = 256
    blocks_per_grid = math.ceil(num_elements / threads_per_block)
    print(f"Launching CUDA kernel with {blocks_per_grid} blocks and {threads_per_block} threads per block.")

    # Launch the kernel
    vector_add_kernel((blocks_per_grid,), (threads_per_block,), (a_gpu, b_gpu, c_gpu, num_elements))

    # Synchronize and transfer result back to host
    cp.cuda.Device().synchronize()  # Ensure all operations are complete
    c_result_host = c_gpu.get()  # Get data from GPU to CPU

    # Verify the result
    expected_result = a_host + b_host
    print(f"Verification successful: {np.allclose(c_result_host, expected_result)}")
    print(f"First 5 elements of C (GPU) : {c_result_host}")
    print(f"First 5 elements of C (Expected) : {expected_result[:5]}")
else:
    print("CUDA is not available. Cannot demonstrate CuPy kernel invocation.")
