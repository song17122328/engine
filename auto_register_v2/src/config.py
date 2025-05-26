import os


class Config:
    # 设置 Flask 的默认时区
    TIMEZONE = "Asia/Shanghai"

    # 如果使用 Celery
    CELERY_TIMEZONE = "Asia/Shanghai"

    # MySQL 时区设置 - 移除 time_zone 参数
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:Yxs2025!@mysql:3306/user_data?charset=utf8mb4'

    # 移除 connect_args 中的时区设置
    SQLALCHEMY_ENGINE_OPTIONS = {
        # 其他引擎选项（如果有）
    }

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'