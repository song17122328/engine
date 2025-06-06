import json
import os
import time
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

CAMPUS_COOKIE_DIR = "campus_cookies"
CAMPUS_URLS = {
    "闵行校区": "https://sjtu.cn/vreg/mh",
    "徐汇校区": "https://sjtu.cn/vreg/xh",
    "七宝校区": "https://sjtu.cn/vreg/qb"
}


def get_cookies_for_campus(campus, times):
    campus_url = CAMPUS_URLS[campus]
    print(f"正在获取 [{campus}] Cookies，访问URL: {campus_url}")

    chrome_options = Options()
    chrome_options.add_argument("--user-agent=Mozilla/5.0...MicroMessenger/8.0...")
    chrome_options.add_argument('--headless')  # 无头模式
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')  # 可选（无头模式推荐）
    chrome_options.add_argument("--window-size=1920,1080")  # Add this line
    service = Service(executable_path='/usr/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(campus_url)
        print("正在访问校区页面...等待8秒")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "iknowbox"))
        )
        time.sleep(8)

        try:
            iknow_checkbox = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "iknow"))
            )
            ActionChains(driver).move_to_element(iknow_checkbox).click().perform()
            print("已点击 iknow 复选框")

            time_radio = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, times))
            )
            time_radio.click()
            print(f"已选择时间: {times}")

            cookies = {c['name']: c['value'] for c in driver.get_cookies()}
            if not is_valid_cookie(cookies):
                raise ValueError("获取的Cookie缺少关键字段")

            print(f"{datetime.now()}: 获取到的Cookies:", cookies)
            save_campus_cookies(campus, cookies)
            return {"status": "success", "message": "已成功获取cookie", "cookies": cookies}

        except Exception as e:
            if "NoSuchElementException" in str(e) or "TimeoutException" in str(e):
                return {"status": "fail", "message": "访客人数已满，官方登记系统已关闭，请等待下一次开始","cookies": None}
            else:
                raise e

    except Exception as e:
        return {"status": "error", "message": f"访问校区页面获取cookies时发生错误: {str(e)}", "cookies": None}
    finally:
        driver.quit()


def is_valid_cookie(cookies):
    """验证Cookie是否完整"""
    required_keys = {'VISITOR', '10.119.6.139:80', 'ik'}
    return all(key in cookies for key in required_keys)


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


def save_campus_cookies(campus, cookies, expiry_hours=2):
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


def get_cookie_path(campus):
    """获取校区对应的Cookie文件路径"""
    os.makedirs(CAMPUS_COOKIE_DIR, exist_ok=True)
    campus_code = CAMPUS_URLS[campus].split('/')[-1]
    return f"{CAMPUS_COOKIE_DIR}/{campus_code}.json"
