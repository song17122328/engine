"""
 `Pipe()` returns a pair of connection objects connected by a pipe. By default, it's duplex (two-way), meaning both ends can send and receive
"""
import multiprocessing
import time


def child_process(conn):
    """Function for the child process using a Pipe connection."""
    print("Child: Waiting to receive data...")
    msg = conn.recv()  # Receive data from the parent
    print(f"Child:Received '{msg}'")

    response = "Hello from Child!"
    print(f"Child: Sending '{response} bach ...'")
    conn.send(response)  # Send response back to the parent
    conn.close()  # Close the connection


if __name__ == '__main__':
    parent_conn, child_conn = multiprocessing.Pipe()  # Create a duplex pipe

    p = multiprocessing.Process(target=child_process, args=(child_conn,))
    p.start()  # Start the child process

    message = "Hello from Parent!"

    print(f"Parent: Sending '{message}' to child...")
    parent_conn.send(message)  # Send data to the child process
    response_from_child = parent_conn.recv()  # Receive response from the child process
    print(f"Parent:Received '{response_from_child}' from child")

    p.join()  # Wait for the child process to finish
    parent_conn.close()  # Close the parent's end of the connection
    print("Main process: Communication finished")