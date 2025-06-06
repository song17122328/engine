#### 1、打开WSL ubuntu终端

#### 2、导航到想要的位置创建项目文件夹：

例如去home建立项目文件夹

```bash
cd ~  # Go to your home directory in WSL2
mkdir my_pytorch_project
cd my_pytorch_project
```

#### 3、创建`.devcontainer`文件夹和`devcontainer.json`文件 

 Inside your `my_pytorch_project` folder, create the `.devcontainer` directory and the `devcontainer.json` file.

```bash
mkdir .devcontainer
touch .devcontainer/devcontainer.json
```

#### 4、在WSL2 项目文件夹中打开VS code:

This is a crucial step. From *within your WSL2 Ubuntu terminal* (while you are inside the `my_pytorch_project` directory), type:

```bash
code .
```

此命令会打开 Visual Studio Code，它直接连接到 WSL2 Ubuntu 文件系统。您会在 Visual Studio Code 窗口的左下角看到 `WSL: Ubuntu`，这表明您正在 WSL 环境中工作

#### 5、**填充 `devcontainer.json`:**

 In the VS Code window that just opened (connected to WSL2), you should see your `my_pytorch_project` folder in the Explorer. Open the `.devcontainer/devcontainer.json` file and paste the following content:

```json
{
    "name": "PyTorch Dev Environment (WSL)",
    "image": "nvcr.io/nvidia/pytorch:24.04-py3",
    "runArgs": ["--gpus", "all"],
    "customizations": {
        "vscode": {
            "settings": {
                // Specify the Python interpreter path within the container
                "python.pythonPath": "/opt/conda/bin/python",
                "python.linting.pylintEnabled": true,
                "python.linting.enabled": true,
                "terminal.integrated.defaultProfile.linux": "bash"
            },
            "extensions": [
                // Recommended extensions for PyTorch development
                "ms-python.python",
                "ms-toolsai.jupyter",
                "ms-vscode.cpptools", // If you need C++ development
                "eamodio.gitlens" // Useful for Git integration
            ]
        }
    },
    // Mount your local project folder into the container.
    // This makes your local code accessible inside the container at /workspace.
    // ${localWorkspaceFolder} correctly resolves to your WSL2 path.
    "mounts": [
        "source=${localWorkspaceFolder},target=/workspace,type=bind"
    ],
    // Set the working directory inside the container
    "workspaceFolder": "/workspace",
    // The user to run as inside the container. NVIDIA images often use 'root'.
    "remoteUser": "root",
    // Optional: Forward a port if your application needs to expose one (e.g., for a web app)
    // "forwardPorts": [8888]
}
```

- **`image`**：指定 Docker 镜像。
- `runArgs`：`--gpus all` 确保 GPU 访问。
- `customizations`：配置 VS Code 设置并在容器内安装扩展。
- `mounts`：这里 `source=${localWorkspaceFolder}`很关键。当VSCode连接到WSL2时，`{localWorkspaceFolder}` 正确指向你 WSL2 文件系统内的路径（例如，`/home/youruser/my_pytorch_project`），然后这个路径会被绑定挂载到容器内的 `/workspace`。
- `workspaceFolder`：设置容器内的默认目录。
- `remoteUser`：指定容器内的用户（对于 NVIDIA 镜像通常为 `root`）。

#### 6、**在容器中重新打开:**

一旦重新保存`devcontainer.json` 文件，VS Code会检测到这一操作，并在右下角弹出一条通知，询问你是否要 “在容器中重新打开”。点击此按钮。

如果没有这条通知，可以手动操作：

- `F1` (or `Ctrl+Shift+P`) 打开命令操作面板.
- 输入`Dev Containers: Reopen in Container` 并选择它.

VS Code 现在将执行以下操作：

* 它将拉取 nvcr.io/nvidia/pytorch:24.04-py3 Docker 镜像（如果尚未缓存）。
* 它将基于此镜像创建并启动一个新的 Docker 容器，并传递 --gpus all 参数。
* 它将把你 WSL2 文件系统中的 my_pytorch_wsl_project 文件夹绑定挂载到容器内的 /workspace 中。
* 它将在容器内安装指定的 VS Code 扩展。
* 最后，它将把 VS Code 连接到这个正在运行的容器。

你会看到左下角的状态从 “WSL: Ubuntu” 变为 “Dev Container: PyTorch Dev Environment (WSL)”。现在你拥有了一个功能完备且可访问 GPU 的 PyTorch 开发环境，并且你的代码文件在 WSL2 项目文件夹与 Docker 容器之间实现了无缝同步。

#### 7、执行命令查看运行情况

* 在VS Code中新建`test.py`

```python
import torch 

print(f"CUDA是否可用:{torch.cuda.is_available()}")
print(f"当前机器GPU数量:{torch.cuda.device_count()}")
print(f"当前GPU设备名:{torch.cuda.get_device_name(0)}")
print(f"当前分布式后端NCCL是否可用:{torch.distributed.is_nccl_available()}")
```

* 保存，点击右上角运行，可以查看下面运行结果。然后就可以愉快的编程了。

![image-20250531142541168](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250531142541168.png)

