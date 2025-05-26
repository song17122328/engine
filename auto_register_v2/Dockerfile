

#version2
# 使用官方 Selenium 镜像（已包含 Chrome + ChromeDriver）
FROM selenium/standalone-chrome:103.0.5060.134-chromedriver-103.0.5060.134-20250515

# 切换到 root 用户操作
USER root

# 安装 Python 3.12 和其他依赖
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    gcc \
    pkg-config \
    default-libmysqlclient-dev \
    tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.12 为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV GUNICORN_TIMEOUT=120
ENV GUNICORN_GRACEFUL_TIMEOUT=120
ENV GUNICORN_WORKERS=1
# 复制项目文件
COPY ./src/requirements.txt .

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt
RUN #pip install --no-cache-dir -r requirements.txt

# 复制整个项目
COPY ./src /app
WORKDIR /app

# 暴露端口
EXPOSE 5000

#CMD ["sh", "-c", "python -c 'from app.tasks.scheduler import run_scheduler; run_scheduler()' & \
#                 exec gunicorn --bind 0.0.0.0:5000 --workers 4 'run:create_app()'"]

# 优化后的启动命令
CMD ["sh", "-c", "python -c 'from app.tasks.scheduler import run_scheduler; run_scheduler()' & \
                 exec gunicorn --bind 0.0.0.0:5000 \
                 --workers $GUNICORN_WORKERS \
                 --timeout $GUNICORN_TIMEOUT \
                 --graceful-timeout $GUNICORN_GRACEFUL_TIMEOUT \
                 'run:create_app()'"]