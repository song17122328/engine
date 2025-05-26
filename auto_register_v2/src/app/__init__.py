from flask import Flask
from .extensions import db, cors
import os  # 添加这一行！
import pytz
import time
from datetime import datetime

from .models import init_db


def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    # 设置 Python 的默认时区
    os.environ['TZ'] = app.config['TIMEZONE']


    app.config['SQLALCHEMY_ECHO'] = True  # 打印所有 SQL 语句
    app.logger.setLevel('DEBUG')  # 启用 DEBUG 日志



    if hasattr(time, 'tzset'):
        time.tzset()

    # 验证时区设置
    # print(f"当前服务器时间: {datetime.now()}")
    # print(f"当前应用时区: {datetime.now(pytz.timezone(app.config['TIMEZONE']))}")

    db.init_app(app)
    cors.init_app(app)

    # 使用相对导入
    from .routes import main_bp
    app.register_blueprint(main_bp)
    # with app.app_context():
    #     print("🚀 执行一次性数据库初始化")
    #     init_db(app)  # 你的初始化函数
    #
    #     print("⏰ 启动定时任务调度器")
    #     scheduler = create_scheduler(app)
    #     scheduler.start()


    return app
