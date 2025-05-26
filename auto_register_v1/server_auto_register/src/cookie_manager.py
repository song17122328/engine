import json
import os
from datetime import datetime, timedelta

CAMPUS_COOKIE_DIR = "campus_cookies"
CAMPUS_URLS = {
    "闵行校区": "https://sjtu.cn/vreg/mh",
    "徐汇校区": "https://sjtu.cn/vreg/xh",
    "七宝校区": "https://sjtu.cn/vreg/qb"
}


def get_cookie_path(campus):
    """获取校区对应的Cookie文件路径"""
    os.makedirs(CAMPUS_COOKIE_DIR, exist_ok=True)
    campus_code = CAMPUS_URLS[campus].split('/')[-1]
    return f"{CAMPUS_COOKIE_DIR}/{campus_code}.json"


def save_campus_cookies(campus, cookies, expiry_hours=24):
    """保存校区特定Cookies"""
    cookie_path = get_cookie_path(campus)
    expiry_time = (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
    data = {
        "campus": campus,
        "cookies": cookies,
        "expiry_time": expiry_time,
        "url": CAMPUS_URLS[campus]
    }
    with open(cookie_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_campus_cookies(campus):
    """加载校区Cookies（检查是否过期）"""
    cookie_path = get_cookie_path(campus)
    if not os.path.exists(cookie_path):
        return None

    with open(cookie_path) as f:
        data = json.load(f)

    if datetime.now() > datetime.fromisoformat(data["expiry_time"]):
        print(f"[{campus}] Cookie已过期")
        return None

    return data["cookies"]


def is_valid_cookie(cookies):
    """验证Cookie是否完整"""
    required_keys = {'VISITOR', '10.119.6.139:80', 'ik'}
    return all(key in cookies for key in required_keys)