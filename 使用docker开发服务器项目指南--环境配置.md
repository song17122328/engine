# 使用docker开发服务器项目指南

### ▌ 为什么要使用docker开发

本地windows的操作系统，服务器Ubuntu操作系统。本地开发配置的环境(库、依赖项)往往在服务器Ubuntu上不适应，故需要docker来确保环境一致性，且课题组服务器多人使用，需要使用docker与其他人的程序进行隔离。

### ▌ 如何在本地(Windows)实现基于服务器Ubuntu的docker开发环境？

配置Windows的 Ubuntu 的docker开发环境：

1）Windows安装WSL2 

WSL2 是 windows subsystem for Linux version2 ，替代了以前的 Hyper-V，可以在Windows 上高效地使用Linux环境，如Ubuntu。

安装WSL2有常见的两种做法：

* 命令操作：对于windows10以上的版本，可以在Windows以管理员身份打开PowerShell 或者CMD，执行 `wsl --install`，系统自动下载WSL2，最为方便。
* 手动下载操作：如果上面的命令操作速度非常慢，可以去https://github.com/microsoft/WSL/releases 选择最新的Assets

![image-20250518111952621](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250518111952621.png)

下载后，直接运行下载的程序，等待安装完毕后，menu菜单会出现WSL Settings，打开如右边所示。

![image-20250518112129238](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250518112129238.png)

然后可以在powershell 使用 `wsl -l -v`   或完整命令`wsl --list --verbose` 查看安装情况，如果出现 `Ubuntu` 则证明安装成功

![image-20250518112507283](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250518112507283.png)

如果WSL Setting安装好了但是没有 Ubuntu ，则在powershell 里面执行 `wsl --install` 自动安装Ubuntu

2）安装decker desktop：从官网上下载docker desktop https://www.docker.com/products/docker-desktop/

![image-20250518112939424](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250518112939424.png)

下载好后直接执行`Docker Desktop installer.exe`, 一路next，最后安装成功

![image-20250518113048307](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250518113048307.png)

3）在WSL2的 Ubuntu环境中启动 docker ，打开WSL settings，按照如下所示

![image-20250518113339402](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250518113339402.png)

### ▌ 在本地Ubuntu中挂代理

具体操作：升级到windows11，Ubuntu会自动加载宿主机的代理

然后在Windows中的C:\Users\<your_username>目录下创建一个.wslconfig文件，然后在文件中写入如下内容，引导wsl的代理到宿主机host

```bash
[experimental]
autoMemoryReclaim=gradual  
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true

```

然后用`wsl --shutdown`关闭WSL，之后再重启

代理设置如下：

![image-20250518144345515](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250518144345515.png)
