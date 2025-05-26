from datetime import datetime

import pytz  # 添加这一行
from flask import Blueprint, jsonify, render_template, request

from .extensions import db
from .services.submission_service import handle_manual_submission

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def home():
    print("进入首页")
    return render_template('index.html')


@main_bp.route('/submit', methods=['POST'])
def submit_form():
    data = request.json
    print(f"表单数据为: {data}")

    required_fields = ['xm', 'zjhm', 'phone', 'campus']
    if not all(field in data for field in required_fields):
        return jsonify({"status": "error", "message": "缺少必要字段"}), 400
    try:
        result = handle_manual_submission(data)
        return result
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"处理表单时出错: {str(e)}"
        }), 500


@main_bp.route('/time')  # 修改为使用蓝图
def get_current_time():
    now = datetime.now()
    shanghai_now = now.astimezone(pytz.timezone('Asia/Shanghai'))

    return f"当前时间（服务器）: {now}<br>当前时间（上海）: {shanghai_now}"


@main_bp.route('/test_db')
def test_db():
    try:
        db.engine.connect()
        return "数据库连接成功!"
    except Exception as e:
        return f"数据库连接失败: {str(e)}"