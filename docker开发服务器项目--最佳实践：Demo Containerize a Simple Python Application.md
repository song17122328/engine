**Demo: Containerize a Simple Python Application**

This demo will create a small Python script and package it into a Docker container.

**Step 1: Create a Project Directory**

Open your terminal (Command Prompt, PowerShell, or your WSL 2 terminal like Ubuntu). Create a new directory for this demo project and navigate into it.

```bash
mkdir simple-docker-demo
cd simple-docker-demo
```

**Step 2: Create the Python Application File (`app.py`)**

Create a file named `app.py` inside the `simple-docker-demo` directory. You can use a text editor (like VS Code, Notepad++, etc.) or your terminal's editor (like `nano` or `vim` if you're in WSL 2).

```python
# app.py
import platform
import sys

print("Hello from inside the Docker container!")
print(f"Running on Python version: {sys.version}")
print(f"Operating System: {platform.system()} {platform.release()}")
print("This script is running in a Docker container on your Windows machine (via WSL2).")
```

**Step 3: Create the Python Requirements File (`requirements.txt`)**

Create a file named `requirements.txt` in the same directory. For this simple script, we don't have any external libraries, but it's standard practice to include this file for Python projects.

```basic
# requirements.txt
# Add your Python dependencies here, one per line.
# Example: flask==2.0.2
```

**Step 4: Create the Dockerfile**

Create a file named `Dockerfile` (with no file extension) in the `simple-docker-demo` directory. This file contains the instructions Docker will follow to build your image.

```dockerfile
# Dockerfile

# Use an official Python runtime as the base image
# We use a specific version (3.9-slim) for stability and smaller size
FROM python:3.9-slim

# Set the working directory inside the container
# All subsequent commands will run from this directory
WORKDIR /app

# Copy the current directory contents (your app code and requirements.txt)
# into the container at the /app directory
COPY . /app

# Install any dependencies specified in requirements.txt
# --no-cache-dir saves space by not caching downloaded packages
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run when the container starts
# This will execute your Python script
CMD ["python", "app.py"]
```

**Step 5: Build the Docker Image**

Open your terminal in the `simple-docker-demo` directory. Run the `docker build` command to build your image using the Dockerfile.

```bash
docker build -t simple-python-app .
```

- `docker build`: The command to initiate an image build.
- `-t simple-python-app`: This tags the resulting image with the name `simple-python-app`. You can use any descriptive name.
- `.`: This tells Docker to look for the `Dockerfile` in the current directory (`.`).

You will see output in your terminal as Docker executes each step in the Dockerfile. It will pull the base image (if not already cached), copy your files, run the `pip install` command, and finally create your image.

**Step 6: Run a Docker Container from Your Image**

Now that you have an image, you can run a container from it.

```bash
docker run simple-python-app
```

**What You Should See:**

When you run the container, you should see the output from your `app.py` script printed in your terminal:

```applescript
Hello from inside the Docker container!
Running on Python version: 3.9.x  # (The exact version from the base image)
Operating System: Linux 5.10.x-microsoft-standard-WSL2 # (Or similar Linux kernel version)
This script is running in a Docker container on your Windows machine (via WSL2).
```

**Congratulations!**

You have successfully:

- Defined a simple application environment using a `Dockerfile`.
- Built a Docker image containing your application and its dependencies.
- Run a container from that image, executing your application in an isolated Linux environment on your Windows machine (thanks to WSL 2).