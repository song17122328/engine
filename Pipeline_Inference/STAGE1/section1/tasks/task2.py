"""
**Next Step:** Expand this to three processes. Process A sends data to B, B processes it and sends its output to C. This simulates a 3-stage pipeline.
"""
import multiprocessing
import time


def process_A(queue_ab):
    for element in range(1, 5):
        numbers = [i * element for i in range(1, 100)]
        print(f"Process A: Generated numbers: {numbers}")
        queue_ab.put(numbers)
        time.sleep(1)  # Simulate some processing time
        print("Process A: Numbers sent to Process B by queue.")
    queue_ab.put(None)


def process_B(queue_ab,queue_bc):
    while True:
        numbers = queue_ab.get()
        if numbers is None:
            queue_bc.put(None)
            break
        result = sum(numbers)
        print(f"Process B: add them and the sum is {result}")
        queue_bc.put(result)


def process_C(queue_bc):
    while True:
        numbers = queue_bc.get()
        if numbers is None:
            break
        numbers /= 2
        print(f"Process C: Divided the sum by 2 and the result is {numbers}")


if __name__ == '__main__':
    queue_ab = multiprocessing.Queue()
    queue_bc = multiprocessing.Queue()
    p_A = multiprocessing.Process(target=process_A, args=(queue_ab,))
    p_B = multiprocessing.Process(target=process_B, args=(queue_ab, queue_bc,))
    p_C = multiprocessing.Process(target=process_C, args=(queue_bc,))

    p_A.start()
    p_B.start()
    p_C.start()

    p_A.join()
    p_B.join()
    p_C.join()

    print("Processes A, B, and C have completed their tasks.")
