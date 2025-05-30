"""
#### Optional socket programming

理解基本的 TCP/IP 和套接字编程为网络通信提供了基础知识，即使像 MPI 或 torch.distributed 这样的高级框架将其抽象化。
套接字允许不同机器（或同一台机器）上的进程通过网络进行通信。
"""

import socket
import threading
import time


# Sverver
def server_program():
    host = socket.gethostname()  # Get the hostname of the server
    port = 12345  # Reserved port for the server
    server_socket = socket.socket()  # Get an instance of the socket
    server_socket.bind((host, port))  # Bind the socket to the host and port

    server_socket.listen(2)  # Listen for incoming connections (max 2 clients)
    print(f"Server is listening on {host}:{port}")

    conn, address = server_socket.accept()  # Accept a connection from a client
    print(f"Connection from {address} has been established.")

    while True:
        data = conn.recv(
            1024).decode()  # Receive data stream from the client. It won't accept data greater than 1024 bytes
        if not data:
            break  # If no data,break
        print(f"Received from client: {data}")
        message = input("-->")
        conn.send(message.encode())  # Send data to the client
    conn.close()  # Close the connection
    server_socket.close()  # Close the server socket


# Client
def client_program():
    host = socket.gethostname()  # As server, use local host
    port = 12345  # The same port as the server

    client_socket = socket.socket()  # Get an instance of the socket
    client_socket.connect((host, port))  # Connect to the server

    message = input(" -> ")  # Take input from the user

    while message.lower().strip() != 'bye':
        client_socket.send(message.encode())  # Send data to the server
        data = client_socket.recv(1024).decode()  # Receive response from the server

        print(f"Client received from server: {data}")  # Print the received data
        message = input(" -> ")  # Take input from the user again

    client_socket.close()  # Close the connection


if __name__ == '__main__':
    # Start the server in a separate thread to keep main thread free
    server_thread = threading.Thread(target=server_program)
    server_thread.daemon = True  # Allow main program to exit even if thread is running
    server_thread.start()

    print("Main: Server started, waiting for client to connect...")
    time.sleep(10000)
    time.sleep(1)  # Give the server some time to start

    client_program()  # Start the client program

    # In a real scenario, you'd manage threads/processes more robustly
    # For this simple example, we'll let the daemon thread die with main.
    print("Main: Client finished. Server thread will terminate.")
# Note: This is a basic example. In production, you would handle exceptions, manage multiple clients, and ensure
# proper shutdown of the server. This code demonstrates a simple client-server communication using sockets in Python.
# The server listens for incoming connections, and the client connects to it to send and receive messages.
