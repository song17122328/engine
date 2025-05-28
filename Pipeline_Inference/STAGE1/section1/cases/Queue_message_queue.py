"""`Queue` provides a way to communicate between processes by allowing them to put and get objects in a thread- and process-safe manner. It's a First-In-First-Out (FIFO) data structure"""

import multiprocessing
import time
import random


def producer(queue):
    """Function to produce data and put it in the queue."""
    print("Producer: 开始生产数据")
    for i in range(5):
        item = random.randint(1, 100)
        queue.put(item)
        print(f"Producer:把 {item} 放入队列")
        time.sleep(random.uniform(0.5, 1.5))
    queue.put(None)  # Sentinel to signal end of production
    print("Producer: 数据生产完成")


def consumer(queue):
    """Function to consume data from the queue."""
    print("Consumer: 开始消费数据")
    while True:
        item = queue.get()
        if item is None:  # Check for sentinel
            break
        print(f"Consumer: 从队列中获取到 {item}")
        time.sleep(random.uniform(0.5, 1.5))
    print("Consumer: 数据消费完成")


if __name__ == '__main__':
    q = multiprocessing.Queue()  # Create a Shared Queue object
    p1=multiprocessing.Process(target=producer, args=(q,))
    p2=multiprocessing.Process(target=consumer, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    print("主进程: 所有任务结束执行")