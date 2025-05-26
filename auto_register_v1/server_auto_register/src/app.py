# app.py
import threading
import time
from datetime import datetime
from datetime import time as dt_time

import requests
from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS
from lxml import html
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import requests
import time
from datetime import datetime
from cookie_manager import *
from proxy import ProxyPool

app = Flask(__name__)
CORS(app)  # 对整个应用启用跨域支持

# 存储用户提交的数据
user_data = {}

# 初始化代理池
proxy_pool = ProxyPool()
# 你的原始提交函数（稍作修改）

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


def get_cookies_for_campus(campus):
    """
    # cookies的一个示例
    cookies = {"ik": "ff3a050529",
               "10.119.6.139:80": "22632.59273.21071.0000",
               "VISITOR": "o3pa76aun638irh8djmtjh3rq6"
               }
    """

    campus_url = CAMPUS_URLS[campus]
    print(f"正在获取 [{campus}] Cookies，访问URL: {campus_url}")
    chrome_options = Options()
    chrome_options.add_argument("--user-agent=Mozilla/5.0...MicroMessenger/8.0...")
    service = Service('./resource/chromedriver.exe')

    driver = webdriver.Chrome(options=chrome_options, service=service)
    try:
        driver.get(campus_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "iknowbox"))
        )
        time.sleep(8)

        # 点击"我已阅读并知晓"复选框
        iknow_checkbox = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "iknow"))
        )
        ActionChains(driver).move_to_element(iknow_checkbox).click().perform()

        # 选择入校时间（示例：选择 time1）
        time_radio = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "time1"))
        )
        time_radio.click()  # 点击选中

        # 获取并验证Cookies
        cookies = {c['name']: c['value'] for c in driver.get_cookies()}
        if not is_valid_cookie(cookies):
            raise ValueError("获取的Cookie缺少关键字段")

        # 打印Cookies
        print(f"{datetime.now()}: 获取到的Cookies:", cookies)
        save_campus_cookies(campus, cookies)
        return cookies

    finally:
        driver.quit()


def submit_application(form_data):
    # 1. 尝试加载现有Cookie
    campus = form_data['campus']
    cookies = load_campus_cookies(campus)
    # 2. 如果没有或无效，则重新获取
    if not cookies or not is_valid_cookie(cookies):
        print(f"{datetime.now()}: 正在获取新Cookie...")
        cookies = get_cookies_for_campus(campus)

    REFERER_URL = "https://qiandao.sjtu.edu.cn/visitor/submit.php"
    # 获取form_data中的校区来决定提交的URL

    proxy = proxy_pool.get_random_proxy()
    request_args = {
        "url": REFERER_URL,
        "data": form_data,
        "headers": headers,
        "timeout": 10,
        "verify": False,  # 关闭证书验证
        "cookies": cookies
    }

    if proxy:
        print(f"使用代理: {proxy}")
        request_args["proxies"] = {"http": proxy, "https": proxy}
    else:
        print("无可用代理IP")

    response = requests.post(**request_args)

    if response.status_code == 200:
        tree = html.fromstring(response.content)
        success_div = tree.xpath('//html/body/div[1]/div[2]/text()')
        print(f"[{datetime.now()}] 提交成功！响应内容:", success_div)
        return True
    else:
        print(f"[{datetime.now()}] 提交失败！HTTP状态码:", response.status_code)
        return False


# 定时任务检查函数
def check_scheduled_tasks():
    while True:
        now = datetime.now()
        # 每天早上8点执行
        if dt_time(16, 15) <= now.time() <= dt_time(23, 50):
            for user_id, data in user_data.items():
                print(f"[{now}] 正在为用户 {user_id} 提交表单...")
                submit_application(data)

        # 每分钟检查一次
        time.sleep(100000)


# 启动定时任务线程
threading.Thread(target=check_scheduled_tasks, daemon=True).start()


@app.route('/')
def home():
    return render_template('index.html')  # 自动从templates目录加载


# 接收前端数据的API
@app.route('/submit', methods=['POST'])
def submit_form():
    data = request.json
    print(f"表单数据为:{data}")

    # 验证必要字段
    required_fields = ['xm', 'zjhm', 'phone']
    if not all(field in data for field in required_fields):
        return jsonify({"status": "error", "message": "缺少必要字段"}), 400

    # 存储用户数据（可以用用户ID或手机号作为key）
    user_id = data['phone']
    user_data[user_id] = data
    # submit_application(data)
    print(user_data)
    return jsonify({
        "status": "success",
        "message": "表单已接收，将在每天早上8点自动提交",
        "user_id": user_id
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
