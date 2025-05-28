"""
Synchronization primitives (locks/events):
虽然在管道示例中没有明确使用（队列和管道处理它们自己的内部同步），但多进程模块也提供了诸如锁（Lock）、事件（Event）、条件（Condition）和信号量（Semaphore）等原语，用于管理对共享资源的访问并协调进程执行，防止竞态条件。
"""
import multiprocessing
import time


def increment_with_lock(shared_value, lock):
    """Worker function that increments a shared value using a lock"""
    for _ in range(5):
        time.sleep(0.01)  # Simulate some work
        lock.acquire()  # Acquire the lock before accessing the shared resource
        try:
            current_val = shared_value.value
            print(f"Process {multiprocessing.current_process().name}: Read {current_val}")
            shared_value.value = current_val + 1
            print(f"Process {multiprocessing.current_process().name}: Wrote {shared_value.value}")
        finally:
            lock.release()  # Ensure the lock is released even if an error occurs


def increment_without_lock(shared_value):
    """Worker function that increments a shared value without a lock (not recommended)"""
    for _ in range(5):
        time.sleep(0.01)  # Simulate some work
        current_val = shared_value.value
        print(f"Process {multiprocessing.current_process().name}: Read {current_val}")
        shared_value.value = current_val + 1
        print(f"Process {multiprocessing.current_process().name}: Wrote {shared_value.value}")


def increment_with_lock_demo():
    """Demonstration of incrementing a shared value with a lock"""
    # Value is a shared memory object , similar to a C type
    shared_int = multiprocessing.Value('i', 0)  # 'i' for signed integer
    lock = multiprocessing.Lock()  # Create a lock for synchronizing access to the shared value
    p1 = multiprocessing.Process(target=increment_with_lock, args=(shared_int, lock), name="P1")
    p2 = multiprocessing.Process(target=increment_with_lock, args=(shared_int, lock), name="P2")

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    print(f"Final value with lock: {shared_int.value}")  # Should be 10 if both processes incremented correctly


def increment_without_lock_demo():
    """Demonstration of incrementing a shared value without a lock (not recommended)"""
    shared_int = multiprocessing.Value('i', 0)  # 'i' for signed integer
    p1 = multiprocessing.Process(target=increment_without_lock, args=(shared_int,), name="P1")
    p2 = multiprocessing.Process(target=increment_without_lock, args=(shared_int,), name="P2")

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    print(f"Final value without lock: {shared_int.value}")  # May not be consistent


if __name__ == '__main__':
    print("Demonstrating increment with lock:")
    increment_with_lock_demo()

    print("\nDemonstrating increment without lock (not recommended):")
    increment_without_lock_demo()

    print("\nSynchronization primitives demonstration completed.")

"""This code demonstrates the use of locks in multiprocessing to safely increment a shared value. The 
`increment_with_lock` function uses a lock to ensure that only one process can modify the shared value at a time, 
preventing"""
