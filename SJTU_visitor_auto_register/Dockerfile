FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖（如需）
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# 复制所有代码
COPY ./src /app

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --upgrade pip

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
