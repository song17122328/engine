wsl2 ubuntu配置代理访问外网：

```bash
# 创建 apt 代理配置文件

echo 'Acquire::http::Proxy "http://127.0.0.1:7890";' | sudo tee /etc/apt/apt.conf.d/proxy.conf
echo 'Acquire::https::Proxy "http://127.0.0.1:7890";' | sudo tee -a /etc/apt/apt.conf.d/proxy.conf
```

但是wsl2配置代理，会影响docker pull，因此可采用下面临时关闭代理，并restart docker desktop

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```