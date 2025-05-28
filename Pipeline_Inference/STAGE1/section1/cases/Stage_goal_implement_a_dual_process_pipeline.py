"""Here, we'll implement a scenario where one process performs a calculation (matrix multiplication) and passes the
results to another process for further action (e.g., printing or logging)."""

import multiprocessing
import numpy as np
import time


def matrix_multiplication(input_queue, output_queue):
    """Process that performs matrix multiplication."""
    print("Matrix Multiplier : Starting...")
    while True:
        matrices = input_queue.get()
        if matrices is None:  # Sentinel value to stop the process
            break
        matrix_a, matrix_b = matrices
        print(f"Matrix Multiplier : Multiplying matrices of shapes {matrix_a.shape}"
              f" and {matrix_b.shape}")
        result = np.dot(matrix_a, matrix_b)
        output_queue.put(result)
        print("Matrix Multiplier : Sent result.")
    print("Matrix Multiplier : Exiting")


def result_consumer(output_queue):
    """Process that consumes and prints the results."""
    print("Result Consumer : Starting...")
    while True:
        result = output_queue.get()
        if result is None:  # Sentinel value to stop the process
            break
        print(f"Result Consumer : Received result with shape {result.shape}")
        print(f"Result Consumer : First 3 x 3 block of result:"
              f"\n{result[:3, :3]}")
    print("Result Consumer : Exiting")


if __name__ == '__main__':
    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()

    # Create and start the matrix multiplication process
    multiplier_process = multiprocessing.Process(target=matrix_multiplication, args=(input_q, output_q))
    consumer_process = multiprocessing.Process(target=result_consumer, args=(output_q,))
    multiplier_process.start()
    consumer_process.start()

    # Generate some random matrices and put them in the input queue
    for i in range(3):
        size = (i + 1) * 100
        mat_a = np.random.rand(size, size)
        mat_b = np.random.rand(size, size)
        print(f"Main Process: Putting matrices (size {size}) into input queue.")
        input_q.put((mat_a, mat_b))
        time.sleep(0.1)  #  some delay
    # Send sentinel values to stop the processes
    input_q.put(None) # Stop the multiplier process


#   等待乘法器完成对其最后一项的处理，并将结果放入输出队列
#   一个更健壮的解决方案可能涉及为消费者设置一个单独的停止信号
#   或者使用可连接队列（JoinableQueue）。为简单起见，我们假设消费者在乘法器完成后结束。
    multiplier_process.join()  # Wait for the multiplier process to finish
    output_q.put(None)  # Stop the consumer process
    consumer_process.join()  # Wait for the consumer process to finish
    print("Main Process: All processes have completed.")



    """
两个Queue对象（input_q和output_q）用于主进程与matrix_multiplier之间的单向通信，然后再从matrix_multiplier到result_consumer。
matrix_multiplier接收矩阵，执行np.dot（矩阵乘法），并将结果放入output_q。
result_consumer检索结果并打印。
None用作哨兵值，以通知进程终止。"""