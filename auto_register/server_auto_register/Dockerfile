# 开发阶段
FROM python:3.9-slim AS dev

WORKDIR /app
COPY ./src/requirements.txt .

RUN pip install --user -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--reload"]

# 生产阶段
FROM python:3.9-slim AS prod

WORKDIR /app
COPY --from=dev /root/.local /root/.local
COPY ./src .

ENV PATH="/root/.local/bin:${PATH}"
RUN pip install gunicorn

CMD ["gunicorn", "-b 0.0.0.0:5000", "--workers=4", "app:app"]
