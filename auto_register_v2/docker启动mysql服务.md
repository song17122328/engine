#### 一、建立mysql的docker服务

1、获取mysql镜像

```bash
docker pull mysql[:tag]
```

**参数说明**

1. **`mysql`**：官方 MySQL 镜像名称
2. **`[:tag]`**（可选）：指定版本标签，不指定时默认使用 `latest`

完整示例命令

```bash
docker pull mysql:8.0
```

2、启动mysql容器

```bash
docker run -d `
	--name mysql_container `
    -p 3306:3306 `
    --network flask_app `
    --network-alias mysql `
    -v flask_app-mysql-data:/var/lib/mysql `
    -e MYSQL_ROOT_PASSWORD='Yxs2025!' `
    -e MYSQL_DATABASE=user_data `
    mysql:8.0
```

**参数解释**

1. **`-d`**
   - 以"分离模式"（detached mode）运行容器，即在后台运行
2. **`--name flask-mysql`**：容器名称
3. **`-p 3306:3306`**：端口映射（主机:容器）
4. **`--network todo-app`**
   - 将容器连接到名为 `todo-app` 的 Docker 网络
   - 这通常用于让多个容器（如应用容器和数据库容器）在同一个网络内通信
5. **`--network-alias mysql`**
   - 为容器设置网络别名为 `mysql`
   - 在同一网络中的其他容器可以通过这个别名访问该 MySQL 容器
6. **`-v todo-mysql-data:/var/lib/mysql`**
   - 设置数据卷映射：
     - `todo-mysql-data`：主机上的卷名（Docker 会自动创建）
     - `/var/lib/mysql`：容器内 MySQL 存储数据的目录
   - 这实现了数据持久化，即使容器删除，数据也会保留
7. **`-e MYSQL_ROOT_PASSWORD='Yxs2025!'`**
   - 设置环境变量，配置 MySQL 的 root 用户密码 Yxs2025!
8. **`-e MYSQL_DATABASE=todos`**
   - 设置环境变量，容器启动时自动创建名为 `todos` 的数据库
9. **`mysql:8.0`**
   - 指定使用 MySQL 镜像, 上面拉取的`mysql:8.0`

服务启动成功

![image-20250522163756205](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250522163756205.png)



3、mysql容器闪退，启动失败，查看日志

`docker logs mysql_container`后发现是版本冲突，之前用过更高版本的mysql，数据保留了下来，现在使用8.0版本被拒绝。

解决办法：1：使用原来高级版本的mysql

​		2：清除旧数据并重新初始化

```bash
# 1. 停止并删除旧容器
docker stop mysql_container
docker rm mysql_container

# 2. 删除旧数据卷
docker volume rm todo-mysql-data

# 3. 重新创建并运行容器
docker run -d `
  --name mysql_container `
  -p 3306:3306 `
  --network todo-app `
  --network-alias mysql `
  -v todo-mysql-data:/var/lib/mysql `
  -e MYSQL_ROOT_PASSWORD='Yxs2025!' `
  -e MYSQL_DATABASE=todos `
  mysql:8.0
```



4、获取容器 ID，以便在下一步中使用。

```bash
docker ps
```

![image-20250522164603220](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250522164603220.png)

5、确认可以连接到 `mysql` 网络上的容器。

运行命令时，输入 `<mysql-container-id>` mysql的容器 ID。

```bash
docker exec -it <mysql-container-id> mysql -p
```

在提示符下，输入创建 `todo-mysql-data` 容器时提供的密码。

![image-20250522164822351](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250522164822351.png)

6、在 MySQL shell 中，列出数据库并验证是否看到 `todos` 数据库。

```sql
SHOW DATABASES;
```

![image-20250522164852815](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250522164852815.png)

#### 二、其他应用连接到mysql

在以下示例中，启动应用并将应用容器连接到 MySQL 容器。

```bash
docker run -dp 3000:3000 `
  --name node_test `
  -w /app -v ${PWD}:/app `
  --network todo-app `
  -e MYSQL_HOST=mysql `
  -e MYSQL_USER=root `
  -e MYSQL_PASSWORD='Yxs2025!' `
  -e MYSQL_DB=todos `
  node:20-alpine `
  sh -c "yarn install && yarn run dev" 
```

**参数详解**	

1. **`-dp 3000:3000`**
   - `-d`：以"分离模式"（detached mode）在后台运行容器
   - `-p 3000:3000`：端口映射，将主机的3000端口映射到容器的3000端口
2. **`--name node_test`** 容器名为：`node_test`
3. **`-w /app`**
   - 设置容器的工作目录为 `/app`，后续命令将在此目录下执行
4. **`-v ${PWD}:/app`**
   - 将当前目录（`${PWD}`）挂载到容器的 `/app` 目录
   - 实现主机和容器之间的文件共享（开发时非常有用）
5. **`--network todo-app`**
   - 将容器连接到名为 `todo-app` 的 Docker 网络
   - 使容器能够访问同一网络中的其他服务（如MySQL）
6. **环境变量设置（数据库连接配置）**
   - `-e MYSQL_HOST=mysql`：指定MySQL主机名为`mysql`（需同一网络中有此别名的容器）
   - `-e MYSQL_USER=root`：使用root用户连接
   - `-e MYSQL_PASSWORD='Yxs2025!'`：设置数据库密码
   - `-e MYSQL_DATABASE=todos`：指定要连接的数据库名
7. **`node:20-alpine`**
   - 使用基于Alpine Linux的Node.js 20镜像
   - Alpine版本体积小，适合生产环境
8. **`sh -c "yarn install && yarn run dev"`**
   - 容器启动后执行的命令：
     1. `yarn install`：安装项目依赖
     2. `yarn run dev`：以开发模式启动应用

##### 完整功能说明

这个命令会：

1. 基于`node:20-alpine`镜像创建并启动一个容器
2. 将当前目录挂载到容器的`/app`目录
3. 设置容器使用`todo-app`网络
4. 配置数据库连接参数
5. 在容器中安装依赖并启动开发服务器
6. 将容器的3000端口暴露给主机的3000端口

![image-20250522170047695](https://cdn.jsdelivr.net/gh/song17122328/MyPic@main/img/image-20250522170047695.png)

执行这个命令后系统会自动拉取`node:20-alpine` 并启动node服务连接mysql