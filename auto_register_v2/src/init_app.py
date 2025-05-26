""" 只运行一次的初始化脚本 """
from app import create_app
from app.models import init_db
from app.tasks.scheduler import create_scheduler

app = create_app()

# 数据库初始化
with app.app_context():
    print("🚀 执行一次性数据库初始化")
    init_db(app)  # 你的初始化函数

    print("⏰ 启动定时任务调度器")
    scheduler = create_scheduler(app)
    scheduler.start()
