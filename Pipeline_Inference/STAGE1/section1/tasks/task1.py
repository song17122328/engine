"""**Specific Task:** Create a simple script where two Python processes communicate. Process A generates a list of
numbers, sends it to Process B using a `multiprocessing.Queue`. Process B receives the list, sums the numbers,
and prints the result."""

import multiprocessing


def process_A(queue):
    """Generates a list of numbers and sends them to B by putting there numbers in the queue"""
    numbers = [i for i in range(1, 100)]
    print(f"Process A: Generated numbers: {numbers}")
    queue.put(numbers)
    print("Process A: Numbers sent to Process B by queue.")


def process_B(queue):
    """Receives the list of numbers from A and sums them"""
    numbers = queue.get()
    print(f"Process B: Received numbers: {numbers}")
    total = sum(numbers)
    print(f"Process B: The sum of the numbers is: {total}")


if __name__ == '__main__':
    data_queue=multiprocessing.Queue()

    p_a=multiprocessing.Process(target=process_A, args=(data_queue,))
    p_b=multiprocessing.Process(target=process_B, args=(data_queue,))

    p_a.start()
    p_b.start()

    p_a.join()
    p_b.join()

    print("Processes A and B have completed their tasks.")
