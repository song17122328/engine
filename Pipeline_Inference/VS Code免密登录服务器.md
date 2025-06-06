### ✅ 第 1 步：生成 SSH 密钥（如果还没有）

打开 Windows 的 **PowerShell** 或 **Git Bash**，运行：

```bash
ssh-keygen -t rsa -b 4096 -C "yuanxiaosong1999@gmail.com"
```

提示输入保存路径时，直接回车（默认路径是：`C:\Users\你的用户名\.ssh\id_rsa`）
 **然后一路回车，不要设置密码。**

------

### ✅ 第 2 步：上传公钥到远程服务器

由于远程使用的是 **非标准端口 15422**，你不能直接用 `ssh-copy-id`，改用以下方式：

```bash
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh -p 15422 yuanxiaosong@202.120.32.244 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

> 上面这条命令建议复制粘贴到 **PowerShell** 中运行。

------

### ✅ 第 3 步：配置 SSH 免密（VS Code 自动识别）

编辑 `C:\Users\你的用户名\.ssh\config` 文件，如果没有就新建一个 `config` 文件（无扩展名），内容如下：

```ssh
Host remote_SJTU
  HostName 202.120.32.244
  Port 15422
  User yuanxiaosong
  IdentityFile ~/.ssh/id_rsa
```

> `remote_SJTU` 是你给远程主机起的名字，**可以自定义**，以后连接就用这个名字。

------

### ✅ 第 4 步：在 VS Code 中连接

1. 打开 VS Code

2. 安装插件：**Remote - SSH**

3. 按 `F1` 或 `Ctrl + Shift + P`，输入并选择：

   ```
   Remote-SSH: Connect to Host...
   ```

4. 选择你刚配置的 `remote202`

✅ 成功后，VS Code 会打开一个远程窗口，自动进入远程主机。**不会再提示输入密码！**

------

## 🧪 测试命令（可选）

你也可以先手动测试连接是否成功：

```bash
ssh remote202
```

如果能直接进服务器，就表示 **配置成功**。

------

## 🛡️ 补充说明

| 项目            | 说明                         |
| --------------- | ---------------------------- |
| 私钥文件        | 千万不要泄露！               |
| 公钥文件 `.pub` | 可以公开，授权登录使用       |
| 默认目录        | `C:\Users\你的用户名\.ssh\`  |
| 支持多个服务器  | 每个服务器加一个 `Host` 就行 |

------

## ✅ 完整文件示例（`~/.ssh/config`）

```ssh
Host remote202
  HostName 202.120.32.244
  Port 15422
  User yuanxiaosong
  IdentityFile ~/.ssh/id_rsa
```

------

如果你在连接过程中遇到任何错误提示（比如 “permission denied” 或 “cannot open file”），截图给我，我来帮你分析并解决。需要我生成一份 `.ssh/config` 文件内容可直接复制用吗？