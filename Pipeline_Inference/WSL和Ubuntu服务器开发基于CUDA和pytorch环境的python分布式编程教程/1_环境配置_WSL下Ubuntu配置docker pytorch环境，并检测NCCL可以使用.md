WSL下Ubuntu配置docker pytorch环境，并检测NCCL可以使用。

#### 1、检查Windows宿主机的CUDA驱动版本

如果Windows宿主机有完整的CUDA环境+CUDA toolkit+pytorch环境，WSL2可以使用到宿主机的CUDA环境，但是CUDA toolkit和pytorch需要重新配置，下面展示在WSL的Ubuntu中配置这些环境，并验证NCCL后端可以使用。

首先进入Ubuntu环境(默认都是WSL2)中使用`Nvidia-smi` 查看CUDA版本，如下图所示`CUDA Version: 12.6` 

**默认情况下toolkit版本等于或小于CUDA版本** ， **高版本的CUDA Driver可以兼容低版本的CUDA Toolkit**

![image-20250529145211365](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250529145211365.png)

如果使用docker环境，可以不需要配置CUDA Toolkit，因为Docker环境里面包括了CUDA Toolkit，只需要确保有CUDA Driver即可，即有上面的面板即可，找到上面的面板对应的CUDA版本直接安装docker对应版本的镜像。

>#### 2、安装WSL2中的 CUDA Toolkit
>
>可以去**访问 [NVIDIA CUDA Toolkit 归档页面](https://developer.nvidia.com/cuda-toolkit-archive) ** 找到对应版本的 **CUDA Toolkit **例如我找到的[CUDA Toolkit 12.63](https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
>
>选择: Linux--> x86_64-->WSL-Ubuntu ( 原生Ubuntu就选Ubuntu ) -->deb(local) ，然后就得到相应安装的Bash命令了。
>
>>1、deb(local)是下载命令包，然后离线解压安装的方式装配到自己电脑上，一次下载，终身受用。
>>
>>2、deb(network) 是通过加入repository到`apt`的sources.list，可以随时从Nvidia服务器上下载最新的package，好处是随时可以更新，但需要经常联网下载，如果访问外网比较好的情况下可以选这个。
>>
>>3、runfile(local) 是一个精简的二进制核心文件，少了很多依赖，不仅可以用在Ubuntu也能用在其他地方，配环境不选它，CUDA开发大佬可能需要。
>
>![image-20250529150230182](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250529150230182.png)
>
>打开Ubuntu WSL2终端，对着上图对应的CUDA bash命令，然后一条一条执行命令即可，第三条命令会下载一个2.8G的文件，比较大。
>
>```bash
>wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
>sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
>wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
>sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.3-1_amd64.deb
>sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
>sudo apt-get update
>sudo apt-get -y install cuda-toolkit-12-6
>```
>
>所有命令执行完成，环境就安装完毕了
>
>##### 把CUDA加入环境变量
>
>```bash
>echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
>source ~/.bashrc
>```
>
>之后通过命令: `nvcc --version` 确认CUDA 所有环境安装完毕
>
>![image-20250529151716499](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250529151716499.png)



#### 3、配置python的docker开发环境

我的是python的12.6的CUDA驱动，高版本兼容低版本，所以我可以使用CUDA12.4的pytorch版本—— `24.04-py3` (CUDA 12.4)

* 拉取pytorch镜像

```bash
docker pull nvcr.io/nvidia/pytorch:24.04-py3
```
环境比较大（20G），经过漫长的等待,终于下载好了

![image-20250529152424209](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250529152424209.png)

#### 4、进入容器，编程

```
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.04-py3
```

- `--gpus all`: 把所有GPU暴露给docker，供开发使用 [必须要有]
- `-it`: 以交互方式进入容器，可以在容器内执行命令 [必须要有]
- `--rm`: 退出容器后，容器自动删除 [可选]
- `nvcr.io/nvidia/pytorch:24.04-py3`: [python和pytorch镜像名]

如下图所示，进入容器内部

![image-20250529152853240](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250529152853240.png)





#### 5、执行命令，检验GPU可用、NCCL可用

1. 检验GPU可用

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

* `print(torch.cuda.is_available())` 打印cuda 能否使用
* `print(torch.cuda.device_count())` 打印GPU数量
* `print(torch.cuda.get_device_name(0))` 打印cuda当前GPU的设备名

如下图所示，这三个输出分别为

`True`——CUDA可用；

`1`——当前有1个GPU；

`NVIDIA GeForce GTX1070`——第一个GPU的设备名：GTX1070,我的老古董GPU了

![image-20250529153201925](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250529153201925.png)

2. 检验NCCL可用

```bash
python -c "import torch; print(torch.distributed.is_nccl_available())"
```
输入上述命令，如下图所示，输出`True`，代表NCCL可用

![image-20250529153526609](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250529153526609.png)