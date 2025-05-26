from datetime import datetime, time

import requests
from flask import current_app
from lxml import html

from .cookie_service import get_cookies_for_campus, is_valid_cookie, load_campus_cookies
from .proxy_service import ProxyPool
from ..extensions import db  # 改为从extensions导入
from ..models import User, Submission

headers = {
    "Cache-Control": "max-age=0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 NetType/WIFI MicroMessenger/7.0.20.1781(0x6701438) WindowsWechat",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://qiandao.sjtu.edu.cn/visitor/",
    "Origin": "https://qiandao.sjtu.edu.cn",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
    "Host": "qiandao.sjtu.edu.cn",
}
# 定义时间范围
start_time = time(8, 0)  # 8:00 AM
end_time = time(20, 0)  # 8:00 PM


def get_time_slot():
    """根据当前时间返回时间段参数"""
    current_hour = datetime.now().hour
    return '1' if 8 <= current_hour < 13 else '2' if 13 <= current_hour < 20 else '0'


def submit_to_system(user, campus):
    """执行实际提交逻辑"""
    time_slot = get_time_slot()

    form_data = {
        'xm': user.xm,
        'zjhm': user.zjhm,
        'phone': user.phone,
        'campus': campus,
        'time': time_slot  # 动态生成的时间参数
    }

    cookies = load_campus_cookies(campus)

    if not cookies or not is_valid_cookie(cookies):

        if start_time <= datetime.now().time() <= end_time:
            print(f"{datetime.now()}: 正在获取新Cookie...")
            cookies = get_cookies_for_campus(campus, "time" + form_data['time'])
        else:
            print("时间不对，不去获取cookies")

    REFERER_URL = "https://qiandao.sjtu.edu.cn/visitor/submit.php"
    # proxy = None
    proxy = ProxyPool().fetch_proxies()

    request_args = {
        "url": REFERER_URL,
        "data": form_data,
        "headers": headers,
        "timeout": 10,
        "verify": False,
        "cookies": cookies
    }

    if proxy:
        current_app.logger.debug(f"使用代理: {proxy}")
        request_args["proxies"] = proxy
    else:
        current_app.logger.debug("无可用代理IP")

    try:
        response = requests.post(**request_args)

        if response.status_code == 200:
            tree = html.fromstring(response.content)
            success_div = tree.xpath('//html/body/div[1]/div[2]/text()')[0]
            # print(f"[{datetime.now()}] 提交成功！响应内容:", success_div)
            if success_div == "登记失败：请通过二维码或公众号进入登记":

                if start_time <= datetime.now().time() <= end_time:
                    get_cookies_for_campus(campus, "time" + form_data['time'])
                    success_div = "cookies过期，已获取新Cookie，请并重新提交"
                else:
                    success_div = "cookies过期，不在获取cookies时间段，请等待8:00-20:00再尝试提交"
            if success_div == "登记失败：您提交登记次数过多":
                success_div = "该用户提交登记次数过多，已被官方系统拉黑"
            return {
                "success": True,
                "message": f"{campus}校区提交成功",
                "time_slot": time_slot,
                "timestamp": datetime.now(),  # 确保包含timestamp
                "success_div": success_div
            }
        else:
            return {
                "success": False,
                "message": f"提交失败，状态码：{response.status_code}",
                "time_slot": time_slot,
                "timestamp": datetime.now(),  # 确保包含timestamp
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"提交异常：{str(e)}",
            "time_slot": time_slot,
            "timestamp": datetime.now(),  # 确保包含timestamp
            "response_text": ""
        }


def handle_manual_submission(form_data):
    """处理手动提交"""
    # 查找或创建用户
    user = User.query.filter_by(phone=form_data['phone']).first()
    if not user:
        user = User(
            xm=form_data['xm'],
            zjhm=form_data['zjhm'],
            phone=form_data['phone']
        )
        db.session.add(user)

    # 查找或创建校区记录
    submission = Submission.query.filter_by(
        user_id=user.id,
        campus=form_data['campus']
    ).first()

    if not submission:
        submission = Submission(
            user_id=user.id,
            campus=form_data['campus']
        )
        db.session.add(submission)

    # 执行提交
    result = submit_to_system(user, form_data['campus'])

    # 更新记录
    submission.last_submitted_at = result['timestamp']
    if result['success']:
        submission.last_result = result['success_div']
    else:
        submission.last_result = result['message']
    db.session.commit()

    return result


def execute_scheduled_submissions():
    """执行定时提交任务"""
    active_submissions = Submission.query.filter_by(is_active=True).join(User).all()

    results = []
    for submission in active_submissions:
        result = submit_to_system(submission.user, submission.campus)

        submission.last_submitted_at = result['timestamp']
        if result['success']:
            submission.last_result = result['success_div']
        else:
            submission.last_result = result['message']
        results.append({
            'user_id': submission.user_id,
            'campus': submission.campus,
            'result': result
        })

    db.session.commit()
    return results
