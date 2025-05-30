"""` To create a new process, you instantiate a `Process` object and pass the target function and its arguments."""
import multiprocessing
import os
import time


def worker_function(name):
    """A function to be executed by a new process."""
    print(f"Worker {name} : 开始执行 (PID :{os.getpid()})")
    time.sleep(2)  # Simulate some work
    print(f"Worker {name} : 结束执行")


if __name__ == '__main__':
    print(f"主进程: 开始执行 (PID :{os.getpid()})")
    # Create a Process Object
    process1 = multiprocessing.Process(target=worker_function, args=("Alice",))
    process2 = multiprocessing.Process(target=worker_function, args=("Bob",))

    # Start the processes
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()
    print(f"主进程: 所有进程结束执行")
