from datetime import datetime

import pytz
from sqlalchemy.ext.declarative import declared_attr

from .extensions import db

# 获取上海时区对象
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')


def shanghai_now():
    """获取当前上海时间"""
    return datetime.now(SHANGHAI_TZ)


def utc_to_shanghai(utc_dt):
    """将 UTC 时间转换为上海时间"""
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    return utc_dt.astimezone(SHANGHAI_TZ)


class BaseModel(db.Model):
    """基础模型类，提供公共字段和方法"""
    __abstract__ = True

    @declared_attr
    def created_at(cls):
        return db.Column(db.DateTime, default=shanghai_now)

    @declared_attr
    def updated_at(cls):
        return db.Column(db.DateTime, default=shanghai_now, onupdate=shanghai_now)

    @property
    def created_at_shanghai(self):
        return utc_to_shanghai(self.created_at).strftime('%Y-%m-%d %H:%M:%S')

    @property
    def updated_at_shanghai(self):
        return utc_to_shanghai(self.updated_at).strftime('%Y-%m-%d %H:%M:%S')


class User(BaseModel):
    """用户基础信息表"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    xm = db.Column(db.String(100), nullable=False)  # 姓名
    zjhm = db.Column(db.String(50), nullable=False)  # 证件号码
    phone = db.Column(db.String(20), nullable=False, unique=True)  # 手机号

    submissions = db.relationship('Submission', back_populates='user', cascade='all, delete-orphan')


class Submission(BaseModel):
    """校区提交记录表"""
    __tablename__ = 'submissions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    campus = db.Column(db.String(50), nullable=False)  # 校区
    last_submitted_at = db.Column(db.DateTime, default=shanghai_now)  # 最后提交时间
    last_result = db.Column(db.String(500))  # 最后一次提交结果
    is_active = db.Column(db.Boolean, default=True)  # 是否活跃记录

    user = db.relationship('User', back_populates='submissions')

    __table_args__ = (
        db.UniqueConstraint('user_id', 'campus', name='_user_campus_uc'),
    )


from sqlalchemy.inspection import inspect

import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
import time


def ensure_database_exists():
    # 从环境变量获取基础连接URI（不带数据库名）
    base_uri = os.getenv('DATABASE_BASE_URI', 'mysql+pymysql://root:password@mysql:3306/')
    db_name = 'user_data'  # 要确保存在的数据库名

    # 第一步：检查数据库是否存在
    admin_engine = create_engine(base_uri)

    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            with admin_engine.connect() as conn:
                # 检查数据库是否存在
                result = conn.execute(text(f"SHOW DATABASES LIKE '{db_name}'"))
                db_exists = result.fetchone() is not None

                if not db_exists:
                    print(f"⏳ 数据库 {db_name} 不存在，尝试创建 (尝试 {attempt + 1}/{max_retries})")
                    # 创建数据库（包含字符集设置）
                    conn.execute(text(f"""
                        CREATE DATABASE `{db_name}`
                        CHARACTER SET utf8mb4
                        COLLATE utf8mb4_unicode_ci
                    """))
                    print(f"✅ 数据库 {db_name} 创建成功")

                # 切换到新数据库验证
                conn.execute(text(f"USE `{db_name}`"))
                print(f"🟢 数据库 {db_name} 已就绪")
                return True

        except OperationalError as e:
            if "Can't connect" in str(e):
                print(f"⌛ MySQL服务未就绪，等待 {retry_delay}秒后重试...")
                time.sleep(retry_delay)
                continue
            raise
        except ProgrammingError as e:
            print(f"❌ 数据库操作失败: {str(e)}")
            raise

    raise RuntimeError(f"无法在 {max_retries} 次尝试内建立数据库连接")


def check_and_create_tables():
    try:
        # 测试连接
        with db.engine.connect() as conn:
            # 设置时区
            conn.execute(text("SET time_zone = '+08:00'"))

            # 获取当前会话的时区
            tz_result = conn.execute(text("SELECT @@session.time_zone"))
            print(f"数据库会话时区: {tz_result.fetchone()[0]}")

            # 获取数据库中已存在的表
            inspector = inspect(db.engine)
            existence_tables = set(inspector.get_table_names())
            print(f"数据库中已存在的表: {existence_tables}")

            tables_to_create = {cls.__tablename__ for cls in BaseModel.__subclasses__()}

            need_to_create = tables_to_create - existence_tables
            print("需要创建的表:", need_to_create)

            # 如果有需要创建的表，则创建
            if need_to_create:
                print(f"开始创建表: {', '.join(need_to_create)}")
                db.create_all()

                # 验证表是否创建成功
                # 使用原生SQL查询表是否存在
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = DATABASE()
                """))
                existence_tables = {row[0] for row in result}
                if existence_tables== need_to_create:
                    print("所有表建立完毕")
                # print(f"直接查询到的表: {existence_tables}")

    except OperationalError as e:
        print(f"数据库操作出错: {e}")
        # 打印详细的错误堆栈
        import traceback
        print(traceback.format_exc())


# 应用启动时自动调用
def init_db(app):
    print("开始初始化数据库")
    with app.app_context():
        ensure_database_exists()
        check_and_create_tables()
