#### 1: Connect VS Code to your Remote Ubuntu Server via SSH

1. **Add your SSH Host (Optional, but Recommended):**
   - Open the Command Palette in VS Code (`F1` or `Ctrl+Shift+P`).
   - Type `Remote-SSH: Add New SSH Host...` and select it.
   - Enter your SSH connection string, e.g., `user@your_server_ip_or_hostname`.
   - VS Code will prompt you where to save the configuration (e.g., `C:\Users\YourUser\.ssh\config` or your WSL2 `~/.ssh/config`). Choose the default or your preferred location.
   - This step makes it easier to connect in the future.
2. **Connect to the Remote Host:**
   - Click on the "Remote Explorer" icon in the VS Code Activity Bar (it looks like a monitor with a plug, or a new window icon).
   - Under the "SSH TARGETS" section, you should see your added host (or you can click the `+` icon to connect directly).
   - Click the "Connect to Host in New Window" icon next to your server's entry.
   - VS Code will open a new window and attempt to connect.
   - It might ask for your password or prompt you to select an SSH key.
   - Once connected, you'll see "SSH: your_server_ip_or_hostname" in the bottom-left corner of the VS Code window.

#### 2: Set up your Project with `devcontainer.json` on the Remote Server

Now that VS Code is connected to your remote server, you'll work with the remote server's filesystem.

1. **Open a Folder on the Remote Server:**

创建`.devcontainer`文件夹和`devcontainer.json`文件 

 Inside your `my_pytorch_project` folder, create the `.devcontainer` directory and the `devcontainer.json` file.

```bash
mkdir my_pytorch_project
cd my_pytorch_project
mkdir .devcontainer
touch .devcontainer/devcontainer.json
```

- In the VS Code window connected to your remote server, go to `File` > `Open Folder...`.

- Navigate to the directory on your remote server where you want your project to reside (e.g., `/home/user/my_remote_pytorch_project`). Click "OK".

- **Populate `devcontainer.json`:**
   The content of your `devcontainer.json` will be almost identical to the one you used for WSL2, as it describes the *Docker container* itself, not the host environment.

   ```json
   {
       "name": "Remote PyTorch Dev Environment",
       //CUDA12.0 对应的版本
       "image": "nvcr.io/nvidia/pytorch:23.12-py3",
       "runArgs": ["--gpus", "all"], // Ensure NVIDIA Container Toolkit is installed on server
       "customizations": {
           "vscode": {
               "settings": {
                   "python.pythonPath": "/opt/conda/bin/python",
                   "python.linting.pylintEnabled": true,
                   "python.linting.enabled": true,
                   "terminal.integrated.defaultProfile.linux": "bash"
               },
               "extensions": [
                   "ms-python.python",
                   "ms-toolsai.jupyter",
                   "ms-vscode.cpptools",
                   "eamodio.gitlens"
               ]
           }
       },
       // Mount the remote project folder into the container.
       // ${localWorkspaceFolder} here refers to the folder opened via SSH on the remote server.
       "mounts": [
           "source=${localWorkspaceFolder},target=/workspace,type=bind"
       ],
       "workspaceFolder": "/workspace",
       "remoteUser": "root",
       // Optional: Forward ports from the container to your local machine
       // "forwardPorts": [8888, 6006] // Example for Jupyter and TensorBoard
   }
   ```

   - **`image`**：指定要使用的 Docker 镜像。
   - runArgs：--gpus all 对于 GPU 访问至关重要。请确保在远程服务器上正确设置了 NVIDIA 容器工具包。
   - mounts：在此上下文中，${localWorkspaceFolder} 指的是你通过 SSH 在远程服务器上打开的文件夹。这确保你在远程服务器上的项目文件被挂载到 Docker 容器中。
   - forwardPorts：这对于远程开发特别有用。如果你的 PyTorch 应用程序运行一个 Web 服务器（例如，用于演示，或者 TensorBoard 在 6006 端口，或者 Jupyter Notebook 在 8888 端口），你可以在此处列出这些端口。VS Code 会自动将它们从远程服务器上的容器转发到你的本地机器，使你能够通过本地浏览器中的[localhost](https://localhost/):PORT 访问它们。

- **Reopen in Container:**

   - Save the `devcontainer.json` file.
   - VS Code will detect the `devcontainer.json` and prompt you with a "Reopen in Container" notification in the bottom-right corner. Click it.
   - If you miss the notification, you can always open the Command Palette (`F1`) and type `Dev Containers: Reopen in Container`.

#### 3: 执行命令查看运行情况

* 在VS Code中新建`test.py`

```python
import torch 

print(f"CUDA是否可用:{torch.cuda.is_available()}")
print(f"当前机器GPU数量:{torch.cuda.device_count()}")
print(f"当前GPU设备名:{torch.cuda.get_device_name(0)}")
print(f"当前分布式后端NCCL是否可用:{torch.distributed.is_nccl_available()}")
```

* 保存，点击右上角运行，可以查看下面运行结果。这台服务器上有8个RTX 3090，然后就可以愉快的编程了。

![image-20250531155251441](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250531155251441.png)
