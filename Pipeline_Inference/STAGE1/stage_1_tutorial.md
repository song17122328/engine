*Thinking...*

> **Analyzing the Outline**
>
> I'm focusing on the initial phase's structure, particularly the Python Distributed Systems section. I'm contemplating the most effective way to introduce core concepts, maybe using practical examples to enhance understanding of these concepts. I need to map out the specifics: data structures, message passing, and fundamental frameworks.
>
> **Mapping the Structure**
>
> I'm currently breaking down the outline of phase one, starting with the introduction to Python Distributed Systems. My focus now is on the 'multiprocessing' module: process creation, the usage of Queues and Pipes. I'm thinking about designing a straightforward example for illustrating these concepts.
>
> **Planning the First Section**
>
> I'm now concentrating on the initial segment of the Python Distributed Systems module, particularly the 'multiprocessing' part. I plan to detail process creation, focusing on both `Queue` and `Pipe` for IPC, and I'm looking for simple yet powerful code examples to bring these concepts to life, including illustrating the implicit serialization using `pickle`. My goal is a dual-process pipeline demonstration.
>
> **Detailing Phase One**
>
> I'm now delving into the first phase, zeroing in on the "Python Distributed System Basics" section, and the `multiprocessing` module. My plan is to begin with `Process` creation, and then introduce `Queue` and `Pipe` functionalities, which will set the stage for a hands-on dual-process pipeline example. This pipeline will feature a calculation performed in one process and passed to the next. I'll explain implicit serialization and introduce `Lock` and `Event`.
>
> **Defining the Curriculum**
>
> I've reviewed the provided outline and started formulating a plan to create a structured course. I'm focusing on creating Python code examples and explanations for both CUDA and distributed systems concepts, as requested. First, I'll explain the `multiprocessing` module in Phase 1, along with demonstrating a dual-process pipeline. I'll then move onto GPU operations in PyTorch.
>
> **Planning the Curriculum**
>
> Now, I'm compiling a detailed curriculum. I'll focus on delivering step-by-step tutorials with Python code, structured in two phases, as requested. Phase 1 will cover `multiprocessing`, CUDA basics using PyTorch, and introduce the concept of parallel computing. Phase 2 will then delve into `torch.distributed` for distributed communication, including an alternative using `mpi4py`. I've begun my research to find accurate and relevant information for all the requested concepts.

The user is asking for a comprehensive, step-by-step tutorial on mastering CUDA and distributed systems, specifically focusing on Python. I will break down the response according to the user's provided outline, including code examples and explanations for each concept.

## Phase 1: Targeted skills acquisition and tool familiarity (fast track)

### 1. Python Distributed System Basics (Concepts and Practice Introduction)

This section focuses on using Python's built-in `multiprocessing` module to simulate distributed environments on a single machine. This helps in understanding fundamental concepts of inter-process communication (IPC) and synchronization without the complexities of network programming or CUDA initially.

#### Focus on mastering the `multiprocessing` module

The `multiprocessing` module in Python allows you to spawn processes, which are independent execution units with their own memory space. This is crucial for parallel computing and understanding distributed systems.

- **Process Creation:**
   To create a new process, you instantiate a `Process` object and pass the target function and its arguments.

  ```python
  import multiprocessing
  import os
  import time
  
  def worker_function(name):
      """A function to be executed by a new process."""
      print(f"Worker {name}: Starting (PID: {os.getpid()})")
      time.sleep(2) # Simulate some work
      print(f"Worker {name}: Finishing")
  
  if __name__ == "__main__":
      print(f"Main process: Starting (PID: {os.getpid()})")
      # Create a Process object
      process1 = multiprocessing.Process(target=worker_function, args=("Alice",))
      process2 = multiprocessing.Process(target=worker_function, args=("Bob",))
  
      # Start the processes
      process1.start()
      process2.start()
  
      # Wait for both processes to complete
      process1.join()
      process2.join()
  
      print("Main process: All workers finished.")
  ```

  *Explanation:*

  - `multiprocessing.Process(target=worker_function, args=("Alice",))` creates a new process that will execute `worker_function` with "Alice" as its argument.
  - `process.start()` begins the execution of the target function in the new process.
  - `process.join()` makes the main process wait until the `process` finishes its execution.

- **`Queue` message queue:**
   `Queue` provides a way to communicate between processes by allowing them to put and get objects in a thread- and process-safe manner. It's a First-In-First-Out (FIFO) data structure. [[1\]](https://www.digitalocean.com/community/tutorials/python-multiprocessing-example)[[2\]](https://superfastpython.com/multiprocessing-queue-in-python/)

  ```python
  import multiprocessing
  import time
  import random
  
  def producer(queue):
      """Produces data and puts it into the queue."""
      print("Producer: Starting")
      for i in range(5):
          item = random.randint(1, 100)
          print(f"Producer: Putting {item} into queue")
          queue.put(item)
          time.sleep(0.5)
      queue.put(None) # Sentinel to signal end of production
      print("Producer: Finished")
  
  def consumer(queue):
      """Consumes data from the queue."""
      print("Consumer: Starting")
      while True:
          item = queue.get()
          if item is None: # Check for sentinel
              break
          print(f"Consumer: Got {item} from queue")
          time.sleep(1) # Simulate processing
      print("Consumer: Finished")
  
  if __name__ == "__main__":
      q = multiprocessing.Queue() # Create a shared queue [26]
  
      p1 = multiprocessing.Process(target=producer, args=(q,))
      p2 = multiprocessing.Process(target=consumer, args=(q,))
  
      p1.start()
      p2.start()
  
      p1.join()
      p2.join()
      print("Main process: All tasks completed.")
  ```

  *Explanation:*

  - `multiprocessing.Queue()` creates a queue that can be shared between processes. [[3\]](https://konfuzio.com/en/python-multiprocessing-queue/)
  - The `producer` process puts items into the queue using `q.put()`.
  - The `consumer` process retrieves items from the queue using `q.get()`.
  - A `None` sentinel value is used to signal the consumer that no more items will be produced, allowing it to terminate gracefully.

- **`Pipe` two-way communication:**
   `Pipe()` returns a pair of connection objects connected by a pipe. By default, it's duplex (two-way), meaning both ends can send and receive. [[4\]](https://docs.python.org/3/library/multiprocessing.html)[[5\]](https://stackoverflow.com/questions/55110733/python-multiprocessing-pipe-communication-between-processes)

  ```python
  import multiprocessing
  import time
  
  def child_process(conn):
      """Function for the child process using a Pipe connection."""
      print("Child: Waiting to receive data...")
      msg = conn.recv() # Receive data from the parent [2]
      print(f"Child: Received '{msg}'")
  
      response = "Hello from Child!"
      print(f"Child: Sending '{response}' back...")
      conn.send(response) # Send data back to the parent [2]
      conn.close() # Close the connection [2]
  
  if __name__ == "__main__":
      parent_conn, child_conn = multiprocessing.Pipe() # Create a duplex pipe [2, 27]
  
      p = multiprocessing.Process(target=child_process, args=(child_conn,))
      p.start()
  
      message = "Hello from Parent!"
      print(f"Parent: Sending '{message}' to child...")
      parent_conn.send(message) # Send data to the child [2]
  
      print("Parent: Waiting to receive response...")
      response_from_child = parent_conn.recv() # Receive data from the child [2]
      print(f"Parent: Received '{response_from_child}'")
  
      p.join()
      parent_conn.close() # Close the parent's end of the connection
      print("Main process: Communication finished.")
  ```

  *Explanation:*

  - `multiprocessing.Pipe()` creates two connection objects, `parent_conn` and `child_conn`. [[4\]](https://docs.python.org/3/library/multiprocessing.html)[[6\]](https://superfastpython.com/multiprocessing-pipe-in-python/)
  - Data sent through one end can be received by the other using `send()` and `recv()`. [[4\]](https://docs.python.org/3/library/multiprocessing.html)[[6\]](https://superfastpython.com/multiprocessing-pipe-in-python/)
  - `conn.close()` is important to release the connection resources.

#### Stage goal: implement a dual-process pipeline

Here, we'll implement a scenario where one process performs a calculation (matrix multiplication) and passes the results to another process for further action (e.g., printing or logging).

```python
import multiprocessing
import numpy as np
import time

def matrix_multiplier(input_queue, output_queue):
    """Process that performs matrix multiplication."""
    print("Matrix Multiplier: Starting...")
    while True:
        matrices = input_queue.get()
        if matrices is None: # Sentinel to stop
            break
        matrix_a, matrix_b = matrices
        print(f"Matrix Multiplier: Multiplying matrices of shapes {matrix_a.shape} and {matrix_b.shape}")
        result = np.dot(matrix_a, matrix_b)
        output_queue.put(result)
        print("Matrix Multiplier: Sent result.")
    print("Matrix Multiplier: Exiting.")

def result_consumer(output_queue):
    """Process that consumes and prints the results."""
    print("Result Consumer: Starting...")
    while True:
        result = output_queue.get()
        if result is None: # Sentinel to stop
            break
        print(f"Result Consumer: Received result matrix of shape {result.shape}")
        # In a real scenario, you might log, store, or further process this result
        print(f"Result Consumer: First 3x3 block of result:\n{result[:3,:3]}")
    print("Result Consumer: Exiting.")

if __name__ == "__main__":
    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()

    # Create and start processes
    multiplier_p = multiprocessing.Process(target=matrix_multiplier, args=(input_q, output_q))
    consumer_p = multiprocessing.Process(target=result_consumer, args=(output_q,))

    multiplier_p.start()
    consumer_p.start()

    # Main process generates tasks
    for i in range(3):
        size = (i + 1) * 100
        mat_a = np.random.rand(size, size)
        mat_b = np.random.rand(size, size)
        print(f"Main: Putting matrices (size {size}) into input queue.")
        input_q.put((mat_a, mat_b))
        time.sleep(0.1) # Small delay

    # Send sentinels to stop the worker processes
    input_q.put(None) # Stop multiplier
    # Wait for multiplier to finish processing its last item and put it to output_q
    # A more robust solution might involve a separate signal for consumer to stop
    # or a JoinableQueue. For simplicity, we'll assume consumer finishes after multiplier.
    multiplier_p.join()
    output_q.put(None) # Stop consumer

    consumer_p.join()
    print("Main: All processes finished.")
```

*Explanation:*

- Two `Queue` objects (`input_q` and `output_q`) are used for unidirectional communication between the main process and `matrix_multiplier`, and then from `matrix_multiplier` to `result_consumer`.
- `matrix_multiplier` receives matrices, performs `np.dot` (matrix multiplication), and puts the result into `output_q`.
- `result_consumer` retrieves the results and prints them.
- `None` is used as a sentinel value to signal processes to terminate.

#### Knowledge gained

- **Process management:** You've learned how to create, start, and wait for processes (`Process`, `start()`, `join()`).

- **Basic IPC (Inter-Process Communication):** You've used `Queue` and `Pipe` to exchange data between independent processes. These mechanisms handle the underlying complexities of moving data across process boundaries. [[4\]](https://docs.python.org/3/library/multiprocessing.html)[[7\]](https://www.educative.io/answers/pipes-queues-and-lock-in-multiprocessing-in-python)

- **Implicit serialization (`pickle`):** When you put Python objects into a `multiprocessing.Queue` or send them through a `Pipe`, Python automatically serializes (pickles) these objects into a byte stream for transmission and deserializes them on the receiving end. [[1\]](https://www.digitalocean.com/community/tutorials/python-multiprocessing-example)[[4\]](https://docs.python.org/3/library/multiprocessing.html) This is why you can pass complex Python objects like NumPy arrays directly.

- **Synchronization primitives (locks/events):** While not explicitly used in the pipeline example (Queues and Pipes handle their own internal synchronization), `multiprocessing` also provides primitives like `Lock`, `Event`, `Condition`, and `Semaphore` to manage access to shared resources and coordinate process execution, preventing race conditions.

   ```python
  import multiprocessing
  import time
  
  def increment_with_lock(shared_value, lock):
      """Worker function that increments a shared value using a lock."""
      for _ in range(5):
          time.sleep(0.01) # Simulate some work
          lock.acquire() # Acquire the lock before accessing shared resource
          try:
              current_value = shared_value.value
              print(f"Process {multiprocessing.current_process().name}: Read {current_value}")
              shared_value.value = current_value + 1
              print(f"Process {multiprocessing.current_process().name}: Wrote {shared_value.value}")
          finally:
              lock.release() # Release the lock
  
  if __name__ == "__main__":
      # Value is a shared memory object, similar to a C type
      shared_int = multiprocessing.Value('i', 0) # 'i' for signed integer
      lock = multiprocessing.Lock() # Create a lock
  
      p1 = multiprocessing.Process(target=increment_with_lock, args=(shared_int, lock), name="P1")
      p2 = multiprocessing.Process(target=increment_with_lock, args=(shared_int, lock), name="P2")
  
      p1.start()
      p2.start()
  
      p1.join()
      p2.join()
  
      print(f"Final shared value: {shared_int.value}")
  ```
  
  Explanation:
  
  - `multiprocessing.Value` creates a shared memory object that processes can directly access.
  - `multiprocessing.Lock()` creates a lock that ensures only one process can access the `shared_int` at a time, preventing race conditions.

#### Optional socket programming

Understanding basic TCP/IP and socket programming provides a foundational knowledge for network communication, even if higher-level frameworks like MPI or `torch.distributed` abstract it away. Sockets allow processes on different machines (or the same machine) to communicate over a network.

```python
import socket
import threading
import time

# --- Server ---
def server_program():
    host = socket.gethostname() # Get local machine name
    port = 12345                # Reserve a port for your service.

    server_socket = socket.socket() # Get instance
    server_socket.bind((host, port)) # Bind to the port

    server_socket.listen(2) # Listen for up to 2 incoming connections
    print(f"Server listening on {host}:{port}")

    conn, address = server_socket.accept() # Accept new connection
    print(f"Connection from: {address}")

    while True:
        data = conn.recv(1024).decode() # Receive data stream. It won't accept more than 1024 bytes
        if not data:
            break # If no data, break
        print(f"Server received: {data}")
        message = input(" -> ")
        conn.send(message.encode()) # Send data to the client

    conn.close() # Close the connection
    server_socket.close()

# --- Client ---
def client_program():
    host = socket.gethostname() # As server, use local host
    port = 12345                # Same port as server

    client_socket = socket.socket() # Instantiate
    client_socket.connect((host, port)) # Connect to the server

    message = input(" -> ") # Take input

    while message.lower().strip() != 'bye':
        client_socket.send(message.encode()) # Send message
        data = client_socket.recv(1024).decode() # Receive response
        print(f"Client received: {data}")
        message = input(" -> ")

    client_socket.close() # Close the connection

if __name__ == "__main__":
    # Run server in a separate thread to keep main process free
    server_thread = threading.Thread(target=server_program)
    server_thread.daemon = True # Allow main program to exit even if thread is running
    server_thread.start()

    time.sleep(1) # Give server a moment to start

    # Run client in the main process
    client_program()

    # In a real scenario, you'd manage threads/processes more robustly
    # For this simple example, we'll let the daemon thread die with main.
    print("Main: Client finished. Server thread will terminate.")
```

*Explanation:*

- **Server:** Creates a socket, binds it to a host and port, listens for incoming connections, accepts a connection, and then sends/receives data.
- **Client:** Creates a socket, connects to the server's host and port, and then sends/receives data.
- `socket.gethostname()` gets the local machine's name. For communication across different machines, you would use their actual IP addresses.
- `send()` and `recv()` methods are used for sending and receiving byte streams. Data needs to be encoded/decoded (e.g., using `.encode()` and `.decode()`) as strings are not directly sent over sockets.

### 2. Learn CUDA through Python (using high-level abstraction)

This section focuses on leveraging GPUs using PyTorch, a high-level deep learning framework that abstracts away much of the underlying CUDA C++ complexities.

#### PyTorch/TensorFlow tensor operations:

- **Master the migration of tensors to GPU (`tensor.to('cuda:0')`):**
   PyTorch tensors can be easily moved between CPU and GPU memory. `cuda:0` refers to the first GPU available. If you have multiple GPUs, you can use `cuda:1`, `cuda:2`, etc.

  ```python
  import torch
  
  # Check if CUDA is available
  if torch.cuda.is_available():
      device = torch.device("cuda:0") # Use the first GPU
      print(f"CUDA is available! Using device: {device}")
  else:
      device = torch.device("cpu")
      print("CUDA not available. Using CPU.")
  
  # Create a tensor on CPU
  cpu_tensor = torch.randn(3, 3)
  print(f"\nCPU Tensor:\n{cpu_tensor}\nDevice: {cpu_tensor.device}")
  
  # Move the tensor to GPU
  if device.type == 'cuda':
      gpu_tensor = cpu_tensor.to(device)
      print(f"\nGPU Tensor:\n{gpu_tensor}\nDevice: {gpu_tensor.device}")
  
      # You can also create a tensor directly on the GPU
      direct_gpu_tensor = torch.zeros(2, 2, device=device)
      print(f"\nDirectly created GPU Tensor:\n{direct_gpu_tensor}\nDevice: {direct_gpu_tensor.device}")
  ```

  *Explanation:*

  - `torch.cuda.is_available()` checks if a CUDA-enabled GPU is present.
  - `torch.device("cuda:0")` specifies the target GPU.
  - `.to(device)` moves the tensor to the specified device.
  - You can also specify the `device` directly when creating a tensor.

- **Basic GPU tensor operations:**
   Once tensors are on the GPU, most PyTorch operations work seamlessly, leveraging the GPU's parallel processing capabilities.

  ```python
  import torch
  
  if torch.cuda.is_available():
      device = torch.device("cuda:0")
      print(f"Performing operations on {device}")
  
      # Create two tensors on GPU
      tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
      tensor_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
  
      print(f"\nTensor A on GPU:\n{tensor_a}")
      print(f"Tensor B on GPU:\n{tensor_b}")
  
      # Element-wise addition
      sum_tensor = tensor_a + tensor_b
      print(f"\nSum (A + B):\n{sum_tensor}\nDevice: {sum_tensor.device}")
  
      # Matrix multiplication
      matmul_tensor = torch.matmul(tensor_a, tensor_b)
      print(f"\nMatrix Multiplication (A @ B):\n{matmul_tensor}\nDevice: {matmul_tensor.device}")
  
      # Scalar multiplication
      scaled_tensor = tensor_a * 2.5
      print(f"\nScaled Tensor (A * 2.5):\n{scaled_tensor}\nDevice: {scaled_tensor.device}")
  
      # Move result back to CPU if needed
      cpu_result = matmul_tensor.to("cpu")
      print(f"\nMatrix Multiplication result moved back to CPU:\n{cpu_result}\nDevice: {cpu_result.device}")
  else:
      print("CUDA not available. Cannot demonstrate GPU tensor operations.")
  ```

  *Explanation:*

  - Operations like `+`, `*`, and `torch.matmul` automatically execute on the GPU if their input tensors are on the GPU.
  - The results of these operations will also reside on the GPU.
  - To use the results on the CPU (e.g., for NumPy conversion or printing without GPU context), you must explicitly move them back using `.to("cpu")`.

- **Understand the synchronization mechanism of `torch.cuda.synchronize()`:**
   CUDA operations are often asynchronous. This means that a PyTorch function that launches a CUDA kernel might return control to the Python script before the GPU computation is actually complete. `torch.cuda.synchronize()` blocks the CPU until all CUDA kernels launched on the current device (or a specific stream) are finished. This is crucial for accurate timing, ensuring data integrity before moving data back to CPU, or debugging.

  ```python
  import torch
  import time
  
  if torch.cuda.is_available():
      device = torch.device("cuda:0")
      print(f"Demonstrating synchronization on {device}")
  
      # Create a large tensor on GPU
      large_tensor = torch.randn(5000, 5000, device=device)
  
      start_time_cpu = time.time()
      # Perform a computationally intensive operation
      result = torch.matmul(large_tensor, large_tensor)
      end_time_cpu = time.time()
      print(f"\nTime taken (CPU timer, without synchronize): {end_time_cpu - start_time_cpu:.4f} seconds")
      # This time might be misleading as the GPU operation might still be running
  
      # --- With synchronization ---
      start_time_sync = time.time()
      result_sync = torch.matmul(large_tensor, large_tensor)
      torch.cuda.synchronize(device) # Wait for all CUDA operations on this device to complete
      end_time_sync = time.time()
      print(f"Time taken (CPU timer, with synchronize): {end_time_sync - start_time_sync:.4f} seconds")
  
      # Example where synchronization is necessary before moving to CPU
      # If you try to access `result.cpu()` before the computation is done,
      # PyTorch will implicitly synchronize, but it's good practice to be aware.
      # Explicit synchronization ensures the data is ready before CPU access.
      cpu_data_ready = result_sync.cpu()
      print(f"Data successfully moved to CPU after synchronization. First element: {cpu_data_ready[0,0]:.4f}")
  
  else:
      print("CUDA not available. Cannot demonstrate synchronization.")
  ```

  *Explanation:*

  - The first `time.time()` measurement might show a very small duration because the Python interpreter quickly queues the GPU operation and moves on, without waiting for the GPU to finish.
  - `torch.cuda.synchronize(device)` forces the CPU to wait until the `matmul` operation on the GPU is truly complete. The second time measurement will thus reflect the actual GPU computation time more accurately.
  - Implicit synchronization occurs when you try to access GPU data on the CPU (e.g., `tensor.cpu()`, `tensor.item()`, `print(tensor)`), but explicit `synchronize()` gives you more control and clarity.

#### Basic CUDA C++ (optional):

This delves into the lower-level aspects of CUDA programming, which is typically done in C++. While PyTorch abstracts this, understanding the basics helps in debugging, optimizing, and writing custom high-performance kernels.

- **Write simple kernel functions (such as vector addition):**
  A CUDA kernel is a function that runs on the GPU. It's defined with the `__global__` specifier. [[8\]](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)[[9\]](http://education.molssi.org/gpu_programming_beginner/02-basic-concepts.html)

  ```cpp
  // vector_add_kernel.cu
  // Compile with: nvcc -shared -o vector_add_kernel.so vector_add_kernel.cu
  
  #include <cuda_runtime.h>
  #include <stdio.h> // For printf in kernel (for debugging, requires specific setup)
  
  // CUDA kernel function for vector addition
  __global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
      // Calculate global index for the current thread
      // Each thread processes one element
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
      if (idx < numElements) {
          C[idx] = A[idx] + B[idx];
      }
  }
  
  // Host function to launch the kernel (optional, for C++ host code)
  extern "C" void launch_vectorAdd(const float* A_d, const float* B_d, float* C_d, int numElements, int blocks, int threadsPerBlock) {
      vectorAdd<<<blocks, threadsPerBlock>>>(A_d, B_d, C_d, numElements);
      cudaDeviceSynchronize(); // Ensure kernel completes before returning
  }
  ```

  *Explanation:*

  - `__global__` indicates that `vectorAdd` is a kernel function that executes on the device (GPU) and can be called from the host (CPU). [[8\]](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)[[9\]](http://education.molssi.org/gpu_programming_beginner/02-basic-concepts.html)
  - `blockIdx.x`, `blockDim.x`, and `threadIdx.x` are built-in CUDA variables used to calculate a unique global index for each thread within the grid. This allows each thread to process a specific element of the vectors.
  - The `if (idx < numElements)` check prevents out-of-bounds access if the number of elements is not perfectly divisible by the total number of threads.

- **Call through PyTorch C++ extension or CuPy:**
  Calling custom CUDA kernels from Python can be done using PyTorch's C++ extensions or libraries like CuPy. CuPy is generally simpler for raw kernel invocation.

  **Using CuPy (simpler for raw kernels):**
   CuPy provides `cupy.RawKernel` to compile and execute CUDA C++ code directly from Python. [[10\]](https://docs.cupy.dev/en/stable/user_guide/kernel.html)[[11\]](https://brainpy.readthedocs.io/en/latest/tutorial_advanced/operator_custom_with_cupy.html)

  ```python
  import cupy as cp
  import numpy as np
  import math
  
  if cp.cuda.is_available():
      # Define the CUDA kernel as a string
      vector_add_kernel_code = r'''
      extern "C" __global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
          int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if (idx < numElements) {
              C[idx] = A[idx] + B[idx];
          }
      }
      '''
  
      # Create a RawKernel object
      # The first argument is the CUDA code, the second is the kernel function name
      vector_add_kernel = cp.RawKernel(vector_add_kernel_code, 'vectorAdd') [6, 18, 33]
  
      # Prepare host data
      num_elements = 1000000
      a_host = np.random.rand(num_elements).astype(np.float32)
      b_host = np.random.rand(num_elements).astype(np.float32)
      c_host = np.zeros(num_elements).astype(np.float32)
  
      # Transfer data to GPU
      a_gpu = cp.asarray(a_host)
      b_gpu = cp.asarray(b_host)
      c_gpu = cp.asarray(c_host)
  
      # Define kernel launch configuration
      threads_per_block = 256
      blocks_per_grid = math.ceil(num_elements / threads_per_block)
  
      print(f"Launching CUDA kernel with {blocks_per_grid} blocks and {threads_per_block} threads per block.")
  
      # Launch the kernel
      # Arguments: (grid_dim, block_dim, (args...))
      vector_add_kernel((blocks_per_grid,), (threads_per_block,), (a_gpu, b_gpu, c_gpu, num_elements)) [6, 18]
  
      # Synchronize and transfer result back to host
      cp.cuda.Device().synchronize() # Ensure kernel completes
      c_result_host = c_gpu.get() # Get data from GPU to CPU
  
      # Verify the result
      expected_result = a_host + b_host
      print(f"Verification successful: {np.allclose(c_result_host, expected_result)}")
      print(f"First 5 elements of C (GPU): {c_result_host[:5]}")
      print(f"First 5 elements of C (Expected): {expected_result[:5]}")
  
  else:
      print("CUDA not available. Cannot demonstrate CuPy kernel execution.")
  ```

  *Explanation:*

  - `cupy.RawKernel` takes the CUDA C++ source code as a string and the name of the kernel function. [[10\]](https://docs.cupy.dev/en/stable/user_guide/kernel.html)[[12\]](https://github.com/cupy/cupy/blob/master/examples/gemm/README.md)
  - `cp.asarray()` transfers NumPy arrays to CuPy arrays (on GPU).
  - The kernel is launched by calling the `vector_add_kernel` object, passing `(grid_dim, block_dim, (args...))`. `grid_dim` and `block_dim` specify the number of thread blocks and threads per block, respectively. [[10\]](https://docs.cupy.dev/en/stable/user_guide/kernel.html)[[12\]](https://github.com/cupy/cupy/blob/master/examples/gemm/README.md)
  - `cp.cuda.Device().synchronize()` ensures the GPU computation is complete before retrieving results.
  - `c_gpu.get()` transfers the CuPy array (on GPU) back to a NumPy array (on CPU).

- **Key points to understand:**

- **Kernel function startup syntax (`<<<>>>`):** This syntax is used in CUDA C++ to launch a kernel on the GPU. It specifies the execution configuration: `kernel_name<<<grid_dim, block_dim, shared_memory_size, stream>>>(args)`. [[13\]](https://stackoverflow.com/questions/51271211/how-is-the-cuda-kernel-launch-syntax-implemented)[[14\]](http://cuda-programming.blogspot.com/2013/01/complete-syntax-of-cuda-kernels.html)

   - `grid_dim`: Defines the number of thread blocks. Can be a 1D, 2D, or 3D dimension.
   - `block_dim`: Defines the number of threads within each block. Can also be 1D, 2D, or 3D.
   - `shared_memory_size` (optional): Amount of dynamic shared memory per block. Defaults to 0.
   - `stream` (optional): CUDA stream to execute the kernel in. Defaults to the default stream.

   - Thread index:

     Inside a kernel, each thread has unique identifiers:

     - `threadIdx.x`, `threadIdx.y`, `threadIdx.z`: Thread ID within its block.
     - `blockIdx.x`, `blockIdx.y`, `blockIdx.z`: Block ID within the grid.
     - `blockDim.x`, `blockDim.y`, `blockDim.z`: Dimensions of the thread block.
     - `gridDim.x`, `gridDim.y`, `gridDim.z`: Dimensions of the grid.
        These are used to calculate a global index for each thread to process specific data elements. For a 1D problem, `global_idx = blockIdx.x * blockDim.x + threadIdx.x`.

   - Memory management (`cudaMalloc`/`cudaMemcpy`):

     In low-level CUDA C++, memory on the GPU (device memory) must be explicitly allocated and deallocated.

     - `cudaMalloc(void** devPtr, size_t size)`: Allocates `size` bytes of memory on the device and returns a pointer to it. [[15\]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
     - `cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)`: Copies `count` bytes between host and device, or device to device. `kind` specifies the direction (e.g., `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`).
     - `cudaFree(void* devPtr)`: Frees memory previously allocated with `cudaMalloc`. [[15\]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
        PyTorch and CuPy handle these operations automatically when you move tensors to/from the GPU (e.g., `tensor.to('cuda')` or `cp.asarray()`), using a caching allocator for efficiency. [[16\]](https://www.codecademy.com/resources/docs/pytorch/gpu-acceleration-with-cuda/memory-management)[[17\]](https://pytorch.org/docs/stable/notes/cuda.html)

## 3. Distributed communication library (Python implementation)

**1. Distributed communication library (Python implementation)**

This phase moves beyond single-machine multiprocessing to actual distributed computing, typically involving multiple GPUs, potentially across multiple machines.

**Highly recommended `torch.distributed`:**

`torch.distributed` is PyTorch's primary library for distributed training and communication. It provides powerful primitives for inter-process communication, optimized for deep learning workloads, especially on GPUs using NCCL. [[18\]](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)[[19\]](https://www.youtube.com/watch?v=reA-ldsTD4Q)

- **Provides MPI-style functions (`send`/`recv`/`broadcast`, etc.):**
   `torch.distributed` offers point-to-point communication (like `send` and `recv`) and collective operations (like `broadcast`, `all_reduce`, `gather`, `scatter`). [[18\]](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)[[20\]](https://brsoff.github.io/tutorials/intermediate/dist_tuto.html)
   To use `torch.distributed`, you first need to initialize the process group. This involves setting environment variables like `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE`.

  ```python
  # This script needs to be run by multiple processes, e.g., using torch.multiprocessing.spawn
  # or a launcher like torchrun (recommended for multi-node).
  
  # Example setup for single-node multi-process (for demonstration):
  # import os
  # os.environ['MASTER_ADDR'] = 'localhost'
  # os.environ['MASTER_PORT'] = '29500' # Choose a free port
  
  import torch
  import torch.distributed as dist
  import torch.multiprocessing as mp
  import os
  
  def run_distributed_example(rank, world_size):
      """Worker function for distributed communication."""
      # Initialize the process group
      # backend='gloo' is good for CPU, 'nccl' for GPU
      dist.init_process_group("gloo", rank=rank, world_size=world_size) [3]
      print(f"Rank {rank}/{world_size}: Process group initialized.")
  
      # --- Point-to-point communication (send/recv) ---
      if rank == 0:
          # Rank 0 sends a tensor to Rank 1
          data_to_send = torch.tensor([10.0, 20.0])
          print(f"Rank {rank}: Sending {data_to_send} to Rank 1")
          dist.send(tensor=data_to_send, dst=1) [1, 3, 34, 39, 43]
      elif rank == 1:
          # Rank 1 receives a tensor from Rank 0
          data_received = torch.zeros(2) # Tensor to receive into
          dist.recv(tensor=data_received, src=0) [1, 3, 34, 39, 43]
          print(f"Rank {rank}: Received {data_received} from Rank 0")
  
      dist.barrier() # Synchronize all processes before next operation
  
      # --- Collective communication (broadcast) ---
      # Rank 0 broadcasts a tensor to all other ranks
      if rank == 0:
          data_to_broadcast = torch.tensor([100.0, 200.0, 300.0])
          print(f"Rank {rank}: Broadcasting {data_to_broadcast}")
      else:
          data_to_broadcast = torch.zeros(3) # Tensor to receive into
  
      dist.broadcast(tensor=data_to_broadcast, src=0) [3, 42]
      print(f"Rank {rank}: Received broadcasted data: {data_to_broadcast}")
  
      dist.barrier()
  
      # --- Collective communication (all_reduce) ---
      # Each rank has a tensor, and all_reduce sums them up across all ranks
      # and makes the sum available on all ranks.
      my_tensor = torch.tensor([float(rank + 1)]) # Each rank has a unique value
      print(f"Rank {rank}: My tensor before all_reduce: {my_tensor}")
      dist.all_reduce(my_tensor, op=dist.ReduceOp.SUM) [3, 35, 36, 42, 47]
      print(f"Rank {rank}: My tensor after all_reduce (SUM): {my_tensor}")
  
      dist.destroy_process_group()
      print(f"Rank {rank}: Process group destroyed.")
  
  def main():
      world_size = 2 # Number of processes
      os.environ['MASTER_ADDR'] = 'localhost'
      os.environ['MASTER_PORT'] = '29500' # Use a random port for robustness
  
      mp.spawn(run_distributed_example,
               args=(world_size,),
               nprocs=world_size,
               join=True)
  
  if __name__ == "__main__":
      main()
  ```

  *Explanation:*

  - `dist.init_process_group()` initializes the communication backend (`gloo` for CPU, `nccl` for GPU). `rank` is the unique ID of the current process, and `world_size` is the total number of processes. [[18\]](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)[[19\]](https://www.youtube.com/watch?v=reA-ldsTD4Q)
  - `dist.send(tensor, dst)` sends a tensor to a specific destination rank. [[21\]](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide)[[22\]](https://github.com/kubeflow/pytorch-operator/blob/master/examples/smoke-dist/dist_sendrecv.py)
  - `dist.recv(tensor, src)` receives a tensor from a specific source rank. [[21\]](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide)[[22\]](https://github.com/kubeflow/pytorch-operator/blob/master/examples/smoke-dist/dist_sendrecv.py)
  - `dist.broadcast(tensor, src)` sends a tensor from the `src` rank to all other ranks in the group. [[18\]](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)[[23\]](https://sagemaker.readthedocs.io/en/v2.66.0/api/training/sdp_versions/v1.0.0/smd_data_parallel_pytorch.html)
  - `dist.all_reduce(tensor, op)` performs a reduction operation (e.g., sum, average) on tensors across all processes, and the result is available on all processes. [[18\]](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)[[24\]](https://discuss.ray.io/t/how-to-get-the-global-loss-to-train-with-pytorch/15401)
  - `dist.barrier()` synchronizes all processes, ensuring all processes reach this point before proceeding.
  - `mp.spawn` is a convenient way to launch multiple processes for distributed training on a single machine. For multi-node setups, `torchrun` (or `mpirun`) is typically used. [[21\]](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide)

- **Supports cross-device GPU communication (using NCCL at the bottom layer):**
   When `backend='nccl'` is used, `torch.distributed` leverages NVIDIA's Collective Communications Library (NCCL) for highly optimized GPU-to-GPU communication. NCCL is designed for high-throughput, low-latency communication between GPUs, especially for operations like `all_reduce` which are critical in distributed deep learning. [[19\]](https://www.youtube.com/watch?v=reA-ldsTD4Q)[[23\]](https://sagemaker.readthedocs.io/en/v2.66.0/api/training/sdp_versions/v1.0.0/smd_data_parallel_pytorch.html) This means data does not need to be transferred to CPU memory before being sent to another GPU.

**Stage goal: Use `torch.distributed` to reconstruct the dual-process pipeline and try multi-GPU deployment**

We'll adapt the matrix multiplication pipeline to use `torch.distributed`, specifically focusing on a multi-GPU scenario if available.

```python
# Save this as, e.g., distributed_pipeline.py
# To run:
# If you have 2 GPUs:
# python -m torch.distributed.launch --nproc_per_node=2 distributed_pipeline.py
# If you have 1 GPU (simulating 2 processes on CPU):
# python -m torch.distributed.launch --nproc_per_node=2 distributed_pipeline.py --backend gloo

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import os
import argparse
import time

def matrix_multiplier_dist(rank, world_size, backend, input_q, output_q):
    """Distributed process that performs matrix multiplication."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500' # Ensure this is consistent or managed by launcher

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized with backend '{backend}'.")

    device = torch.device(f"cuda:{rank}" if backend == "nccl" and torch.cuda.is_available() else "cpu")
    print(f"Rank {rank}: Using device {device}")

    if rank == 0: # This process will act as the "multiplier"
        print(f"Rank {rank}: Acting as Matrix Multiplier.")
        while True:
            # Receive matrices from "input_q" (conceptually, could be from another process)
            # For simplicity, we'll simulate input directly or from a queue if needed.
            # In a real DDP scenario, data loading would be handled by a DistributedSampler.
            matrices_data = input_q.get() # This queue is for inter-process communication on CPU
            if matrices_data is None:
                print(f"Rank {rank}: Received stop signal.")
                break

            matrix_a_np, matrix_b_np = matrices_data
            matrix_a = torch.from_numpy(matrix_a_np).to(device)
            matrix_b = torch.from_numpy(matrix_b_np).to(device)

            print(f"Rank {rank}: Multiplying matrices on {device} of shapes {matrix_a.shape} and {matrix_b.shape}")
            result = torch.matmul(matrix_a, matrix_b)

            # Send the result to Rank 1 (consumer)
            print(f"Rank {rank}: Sending result to Rank 1.")
            dist.send(tensor=result.cpu(), dst=1) # Send CPU tensor to avoid cross-GPU direct send for simplicity here

        dist.send(tensor=torch.tensor([-1.0]), dst=1) # Send a sentinel to consumer
    elif rank == 1: # This process will act as the "consumer"
        print(f"Rank {rank}: Acting as Result Consumer.")
        while True:
            received_tensor = torch.zeros(100, 100, device=device) # Pre-allocate for expected size
            # A more robust solution would involve sending size first, or using a known size.
            # For this example, we assume results are 100x100 for the first few iterations.

            dist.recv(tensor=received_tensor, src=0)
            if received_tensor.item() == -1.0: # Check for sentinel
                print(f"Rank {rank}: Received stop signal.")
                break

            print(f"Rank {rank}: Received result matrix on {device} of shape {received_tensor.shape}")
            print(f"Rank {rank}: First 3x3 block of result:\n{received_tensor[:3,:3].cpu()}")

    dist.destroy_process_group()
    print(f"Rank {rank}: Process group destroyed.")

def main_launcher(backend):
    world_size = 2 # Fixed for this example
    input_q_cpu = mp.Queue() # Queue to send initial data from main to rank 0
    output_q_cpu = mp.Queue() # Not directly used in this DDP example, but kept for consistency

    # Start the processes using torch.multiprocessing.spawn
    # In a real multi-node setup, you'd use torchrun or mpirun
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=matrix_multiplier_dist, args=(rank, world_size, backend, input_q_cpu, output_q_cpu))
        p.start()
        processes.append(p)

    # Main process (which is separate from the spawned distributed processes)
    # This simulates data generation or loading
    if world_size > 0: # Only generate data if there's a rank 0 to receive it
        for i in range(3):
            size = 100 # Fixed size for simplicity in this example
            mat_a = np.random.rand(size, size).astype(np.float32)
            mat_b = np.random.rand(size, size).astype(np.float32)
            print(f"Main Launcher: Putting matrices (size {size}) into input queue for Rank 0.")
            input_q_cpu.put((mat_a, mat_b))
            time.sleep(0.1)

        input_q_cpu.put(None) # Signal Rank 0 to stop

    for p in processes:
        p.join()
    print("Main Launcher: All distributed processes finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="gloo", choices=['gloo', 'nccl'],
                        help="Backend for distributed communication (gloo for CPU, nccl for GPU).")
    args = parser.parse_args()

    # If using nccl, ensure CUDA is available
    if args.backend == "nccl" and not torch.cuda.is_available():
        print("Warning: NCCL backend requested but CUDA is not available. Falling back to Gloo.")
        args.backend = "gloo"

    # This example uses mp.spawn for simplicity of demonstration within one script.
    # For actual multi-GPU/multi-node deployments, `torch.distributed.launch` or `torchrun`
    # is the standard way to set up the environment variables (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE)
    # and launch the processes.
    main_launcher(args.backend)
```

*Explanation:*

- The script is designed to be run with `torch.distributed.launch` or `torchrun`, which automatically sets up environment variables like `RANK` and `WORLD_SIZE` for each spawned process.
- Each process (`rank`) initializes its own distributed environment.
- `rank 0` acts as the matrix multiplier, receiving data (simulated via `input_q_cpu` from the `main_launcher` process), performing multiplication on the designated device (GPU if `nccl` and available, else CPU), and sending the result to `rank 1`.
- `rank 1` acts as the result consumer, receiving the computed matrix from `rank 0`.
- Data transfer between processes uses `dist.send` and `dist.recv`. Note that `tensor.cpu()` is used before sending to simplify the example, but with NCCL, direct GPU-to-GPU transfers are highly optimized.
- This setup demonstrates how `torch.distributed` can manage communication for a pipeline across different processes, potentially leveraging multiple GPUs.

**Alternative solution `mpi4py`:**

`mpi4py` provides Python bindings for the Message Passing Interface (MPI) standard. MPI is a widely used standard for parallel programming on distributed-memory systems. `mpi4py` is suitable when you need more low-level control over communication patterns or when integrating with existing MPI-based HPC (High-Performance Computing) environments. [[25\]](https://education.molssi.org/parallel-programming/03-distributed-examples-mpi4py.html)[[26\]](https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf)

```python
# Save this as, e.g., mpi_example.py
# To run: mpirun -np 2 python mpi_example.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD # Get the global communicator [9, 22]
rank = comm.Get_rank() # Get the rank of the current process [9, 22]
size = comm.Get_size() # Get the total number of processes [9, 22]

print(f"Hello from process {rank} of {size}")

if rank == 0:
    # Rank 0 (master) sends data
    data_to_send = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    print(f"Rank {rank}: Sending {data_to_send} to Rank 1")
    comm.send(data_to_send, dest=1, tag=11) # Send Python object [7, 22, 37]
    # For NumPy arrays, you can use the faster Send/Recv with buffer protocol
    # comm.Send([data_to_send, MPI.FLOAT], dest=1, tag=11) [28, 37]

    # Rank 0 also receives from Rank 1
    received_data_from_1 = comm.recv(source=1, tag=12)
    print(f"Rank {rank}: Received {received_data_from_1} from Rank 1")

elif rank == 1:
    # Rank 1 (worker) receives data
    received_data = comm.recv(source=0, tag=11) # Receive Python object [7, 22, 37]
    # For NumPy arrays, you can use the faster Send/Recv with buffer protocol
    # data_buffer = np.empty(3, dtype=np.float32)
    # comm.Recv([data_buffer, MPI.FLOAT], source=0, tag=11) [15, 28, 37]
    print(f"Rank {rank}: Received {received_data} from Rank 0")

    # Rank 1 sends data back to Rank 0
    response_data = np.array([4.0, 5.0], dtype=np.float32)
    print(f"Rank {rank}: Sending {response_data} to Rank 0")
    comm.send(response_data, dest=0, tag=12)

# --- Collective Communication (Broadcast) ---
# Rank 0 broadcasts a value to all other ranks
if rank == 0:
    bcast_data = {'value': 123}
    print(f"Rank {rank}: Broadcasting {bcast_data}")
else:
    bcast_data = None # Will be filled by broadcast

bcast_data = comm.bcast(bcast_data, root=0) # Broadcast Python object [37]
print(f"Rank {rank}: Received broadcasted data: {bcast_data}")

# --- Collective Communication (Allreduce) ---
# Each process contributes a value, and the sum is available on all processes
my_value = rank + 1
sum_all_ranks = comm.allreduce(my_value, op=MPI.SUM) [46]
print(f"Rank {rank}: My value: {my_value}, Sum of all ranks: {sum_all_ranks}")
```

*Explanation:*

- `MPI.COMM_WORLD` is the default communicator that includes all processes. [[25\]](https://education.molssi.org/parallel-programming/03-distributed-examples-mpi4py.html)[[26\]](https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf)
- `comm.Get_rank()` and `comm.Get_size()` retrieve the unique ID of the current process and the total number of processes, respectively. [[25\]](https://education.molssi.org/parallel-programming/03-distributed-examples-mpi4py.html)[[26\]](https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf)
- `comm.send()` and `comm.recv()` are used for point-to-point communication of Python objects (which are pickled). [[26\]](https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf)[[27\]](https://pythonprogramming.net/sending-receiving-data-messages-mpi4py/)
- `comm.bcast()` broadcasts an object from the `root` process to all other processes. [[28\]](https://mpi4py.readthedocs.io/en/4.0.3/tutorial.html)
- `comm.allreduce()` performs a reduction operation (like `MPI.SUM`) on values from all processes, with the result available on all processes. [[29\]](https://docs.ycrc.yale.edu/clusters-at-yale/guides/mpi4py/)
- For better performance with NumPy arrays, `mpi4py` also provides uppercase `Send`/`Recv` methods that use MPI's buffer protocol, avoiding Python's pickling overhead. [[28\]](https://mpi4py.readthedocs.io/en/4.0.3/tutorial.html)[[30\]](https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/)

This comprehensive tutorial covers the core concepts and practical implementations for Python distributed systems and CUDA programming, moving from single-machine simulations to multi-process/multi-GPU distributed environments.

------

Learn more:

1. [Python Multiprocessing Example - DigitalOcean](https://www.digitalocean.com/community/tutorials/python-multiprocessing-example)
2. [Multiprocessing Queue in Python](https://superfastpython.com/multiprocessing-queue-in-python/)
3. [Python multiprocessing queue for efficient data management - Konfuzio](https://konfuzio.com/en/python-multiprocessing-queue/)
4. [multiprocessing — Process-based parallelism — Python 3.13.3 documentation](https://docs.python.org/3/library/multiprocessing.html)
5. [Python multiprocessing - pipe communication between processes - Stack Overflow](https://stackoverflow.com/questions/55110733/python-multiprocessing-pipe-communication-between-processes)
6. [Multiprocessing Pipe in Python](https://superfastpython.com/multiprocessing-pipe-in-python/)
7. [Pipes, queues, and lock in multiprocessing in Python - Educative.io](https://www.educative.io/answers/pipes-queues-and-lock-in-multiprocessing-in-python)
8. [Tutorial 01: Say Hello to CUDA](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
9. [Basic Concepts in CUDA Programming - MolSSI Education](http://education.molssi.org/gpu_programming_beginner/02-basic-concepts.html)
10. [User-Defined Kernels — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/kernel.html)
11. [GPU Operator Customization with CuPy - BrainPy documentation - Read the Docs](https://brainpy.readthedocs.io/en/latest/tutorial_advanced/operator_custom_with_cupy.html)
12. [cupy/examples/gemm/README.md at main - GitHub](https://github.com/cupy/cupy/blob/master/examples/gemm/README.md)
13. [How is the CUDA<<<...>>>() kernel launch syntax implemented - Stack Overflow](https://stackoverflow.com/questions/51271211/how-is-the-cuda-kernel-launch-syntax-implemented)
14. [Complete syntax of CUDA Kernels](http://cuda-programming.blogspot.com/2013/01/complete-syntax-of-cuda-kernels.html)
15. [CUDA Runtime API - 6.11. Memory Management - NVIDIA Docs Hub](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
16. [PyTorch | GPU Acceleration with CUDA | Memory Management - Codecademy](https://www.codecademy.com/resources/docs/pytorch/gpu-acceleration-with-cuda/memory-management)
17. [CUDA semantics — PyTorch 2.7 documentation](https://pytorch.org/docs/stable/notes/cuda.html)
18. [Writing Distributed Applications with PyTorch](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)
19. [03. Intro to torch.distributed - YouTube](https://www.youtube.com/watch?v=reA-ldsTD4Q)
20. [Writing Distributed Applications with PyTorch](https://brsoff.github.io/tutorials/intermediate/dist_tuto.html)
21. [Multi node PyTorch Distributed Training Guide For People In A Hurry - Lambda](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide)
22. [pytorch-operator/examples/smoke-dist/dist_sendrecv.py at master - GitHub](https://github.com/kubeflow/pytorch-operator/blob/master/examples/smoke-dist/dist_sendrecv.py)
23. [PyTorch Guide to SageMaker's distributed data parallel library](https://sagemaker.readthedocs.io/en/v2.66.0/api/training/sdp_versions/v1.0.0/smd_data_parallel_pytorch.html)
24. [How to get the global loss to train with pytorch? - Ray](https://discuss.ray.io/t/how-to-get-the-global-loss-to-train-with-pytorch/15401)
25. [MPI Hands-On - mpi4py — Parallel Programming | MolSSI Education documentation](https://education.molssi.org/parallel-programming/03-distributed-examples-mpi4py.html)
26. [Parallel Computing in Python using mpi4py](https://research.computing.yale.edu/sites/default/files/files/mpi4py.pdf)
27. [Sending and Receiving data using send and recv commands with MPI - PythonProgramming.net](https://pythonprogramming.net/sending-receiving-data-messages-mpi4py/)
28. [Tutorial — MPI for Python 4.0.3 documentation](https://mpi4py.readthedocs.io/en/4.0.3/tutorial.html)
29. [MPI Parallelism with Python - Yale Center for Research Computing](https://docs.ycrc.yale.edu/clusters-at-yale/guides/mpi4py/)
30. [Parallel programming in Python: mpi4py (part 2) – PDC Blog - KTH](https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/)