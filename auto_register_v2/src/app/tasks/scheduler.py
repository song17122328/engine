import os
import time
from datetime import datetime
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from ..extensions import db
from ..models import init_db
from ..services.submission_service import execute_scheduled_submissions

# 全局应用实例
temp_app = None


def init_database():
    """独立初始化数据库"""
    global temp_app
    db_url = os.getenv('DATABASE_URI')

    if not db_url:
        raise ValueError("未找到数据库配置！请设置 DATABASE_URL 环境变量")

    temp_app = Flask(__name__)
    temp_app.config.update({
        'SQLALCHEMY_DATABASE_URI': db_url,
        'SQLALCHEMY_TRACK_MODIFICATIONS': False
    })

    db.init_app(temp_app)
    with temp_app.app_context():
        init_db(temp_app)
        print(f"✅ 数据库初始化完成（连接: {db_url.split('@')[-1]}）")
    return temp_app


def job_wrapper():
    """包装任务函数"""
    global temp_app
    with temp_app.app_context():
        try:
            print(f"[{datetime.now()}] 执行定时任务")
            result = execute_scheduled_submissions()
            print(result)
            db.session.commit()
        except Exception as e:
            print(f"任务执行失败: {str(e)}")
            db.session.rollback()
        finally:
            db.session.remove()


def run_scheduler():
    """启动调度器"""
    init_database()

    scheduler = BackgroundScheduler()
    # 修正：传递函数引用而非调用结果
    scheduler.add_job(job_wrapper, 'cron', hour=7, minute=0, id='morning_submission')
    scheduler.add_job(job_wrapper, 'cron', hour=13, minute=0,id='afternoon_submission')

    scheduler.start()
    print("⏰ 定时任务调度器已启动")

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
