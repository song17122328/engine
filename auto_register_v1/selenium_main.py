from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
from datetime import datetime
import schedule
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

service = Service('server_auto_register/resource/chromedriver.exe')


def submit_application(t="time1"):
    chrome_options = Options()
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36 MicroMessenger/8.0.25.2140(0x28001937)")
    chrome_options.add_argument("--disable-web-security")  # 允许跨域
    # chrome_options.add_argument("--headless")  # 无头模式（可选）
    # 启动浏览器
    chromedriver_path = "server_auto_register/resource/chromedriver.exe"  # 替换为你的chromedriver路径
    driver = webdriver.Chrome(options=chrome_options, service=service)

    # 访问目标页面
    driver.get("https://sjtu.cn/vreg/mh")  #闵行校区
    # driver.get("https://sjtu.cn/vreg/xh")#徐汇校区
    print(f"{datetime.now()}: 已打开页面，标题: {driver.title}")

    # 检查是否成功进入
    print("当前页面标题:", driver.title)

    # # 等待页面加载
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "iknowbox"))
    )

    # 等待阅读倒计时结束 (8秒)
    time.sleep(8)
    # 点击"我已阅读并知晓"复选框
    iknow_checkbox = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "iknow"))
    )
    ActionChains(driver).move_to_element(iknow_checkbox).click().perform()

    # 选择入校时间（示例：选择 time2）
    time2_radio = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, t))
    )
    time2_radio.click()  # 点击选中

    cookies = driver.get_cookies()
    # 这里打印完整的cookies，包括VISITOR 10.119.6.139:80 ik
    print(cookies)
    time.sleep(20000)
    # 填写表单
    # 1. 姓名
    name_field = driver.find_element(By.ID, "xm")
    name_field.clear()
    name_field.send_keys("张珩勋")

    time.sleep(2)  # 等待2秒，避免操作过快
    # 2. 证件号码
    id_field = driver.find_element(By.ID, "zjhm")
    id_field.clear()
    id_field.send_keys("460001199403270711")  # 替换为你的证件号

    time.sleep(2)  # 等待2秒，避免操作过快
    # 3. 手机号
    phone_field = driver.find_element(By.ID, "phone")
    phone_field.clear()
    phone_field.send_keys("17821696076")  # 替换为你的手机号

    time.sleep(3)  # 等待2秒，避免操作过快
    time.sleep(200000)
    # 点击登记按钮
    submit_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "submitbtn"))
    )
    submit_btn.click()

    # 等待提交完成
    time.sleep(3)
    if t == "time1":
        print(f"{datetime.now()}: 早上8：00 申请提交成功")
    else:
        print(f"{datetime.now()}: 下午13：00 申请提交成功")

    driver.quit()


def doDouble():
    submit_application("time1")
    submit_application("time2")


def scheduleDouble():
    # 设置定时任务
    schedule.every().day.at("8:01").do(doDouble)  # 早上8点
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    # scheduleDouble()
    submit_application("time2")
