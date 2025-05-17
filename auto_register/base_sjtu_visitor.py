import os
import sys
import time
from datetime import datetime
import PySimpleGUI as sg
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service  # 新增导入
import threading

# 设置主题
sg.theme('LightBlue2')


def submit_application(values, window):
    try:
        # 从界面获取参数
        name = values['-NAME-']
        id_num = values['-ID-']
        phone = values['-PHONE-']
        time_choice = values['-TIME-']  # 'time1'或'time2'

        # 更新日志
        window['-LOG-'].print(f"{datetime.now()}: 开始执行自动化...")

        chrome_options = Options()
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36 MicroMessenger/8.0.25.2140(0x28001937)")
        chrome_options.add_argument("--disable-web-security")

        # 获取当前脚本所在目录
        if getattr(sys, 'frozen', False):
            # 打包后的exe所在目录
            application_path = os.path.dirname(sys.executable)
        else:
            # 脚本所在目录
            application_path = os.path.dirname(os.path.abspath(__file__))

        # 设置ChromeDriver路径（新版Selenium使用Service对象）
        chromedriver_path = os.path.join(application_path, 'chromedriver.exe')
        service = Service(executable_path=chromedriver_path)  # 创建Service对象
        driver = webdriver.Chrome(service=service, options=chrome_options)  # 传入service参数

        window['-LOG-'].print(f"{datetime.now()}: 已打开浏览器")

        # 访问目标页面
        driver.get("https://sjtu.cn/vreg/mh")
        window['-LOG-'].print(f"{datetime.now()}: 已打开页面，标题: {driver.title}")

        # 等待页面加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "iknowbox"))
        )

        # 等待阅读倒计时结束 (8秒)
        for i in range(8, 0, -1):
            window['-LOG-'].print(f"请等待阅读倒计时: {i}秒")
            time.sleep(1)

        # 点击"我已阅读并知晓"复选框
        iknow_checkbox = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "iknow"))
        )
        ActionChains(driver).move_to_element(iknow_checkbox).click().perform()
        window['-LOG-'].print(f"{datetime.now()}: 已勾选同意复选框")

        # 选择入校时间
        time_radio = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, time_choice))
        )
        time_radio.click()
        window['-LOG-'].print(
            f"{datetime.now()}: 已选择入校时间段 {'7:00-13:00' if time_choice == 'time1' else '13:00-20:00'}")

        # 填写表单
        name_field = driver.find_element(By.ID, "xm")
        name_field.clear()
        name_field.send_keys(name)
        window['-LOG-'].print(f"{datetime.now()}: 已填写姓名")

        id_field = driver.find_element(By.ID, "zjhm")
        id_field.clear()
        id_field.send_keys(id_num)
        window['-LOG-'].print(f"{datetime.now()}: 已填写证件号")

        phone_field = driver.find_element(By.ID, "phone")
        phone_field.clear()
        phone_field.send_keys(phone)
        window['-LOG-'].print(f"{datetime.now()}: 已填写手机号")

        # 点击登记按钮
        submit_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "submitbtn"))
        )
        submit_btn.click()
        window['-LOG-'].print(f"{datetime.now()}: 已提交登记")

        # 等待提交完成
        time.sleep(3)
        window['-LOG-'].print(f"{datetime.now()}: 登记成功！")

        # 截图保存
        screenshot_path = os.path.join(application_path, f"success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        driver.save_screenshot(screenshot_path)
        window['-LOG-'].print(f"截图已保存到: {screenshot_path}")

    except Exception as e:
        window['-LOG-'].print(f"{datetime.now()}: 发生错误 - {str(e)}")
        # 错误截图
        if 'driver' in locals():
            screenshot_path = os.path.join(application_path, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            driver.save_screenshot(screenshot_path)
            window['-LOG-'].print(f"错误截图已保存到: {screenshot_path}")
    finally:
        if 'driver' in locals():
            driver.quit()
            window['-LOG-'].print(f"{datetime.now()}: 浏览器已关闭")


def create_window():
    # 布局设计
    layout = [
        [sg.Text('姓名:'), sg.InputText(key='-NAME-', size=(30, 1))],
        [sg.Text('证件号:'), sg.InputText(key='-ID-', size=(30, 1))],
        [sg.Text('手机号:'), sg.InputText(key='-PHONE-', size=(30, 1))],
        [sg.Text('入校时间:'),
         sg.Radio('7:00-13:00', "TIME", default=True, key='time1'),
         sg.Radio('13:00-20:00', "TIME", key='time2')],
        [sg.Button('开始登记', key='-SUBMIT-'), sg.Button('退出')],
        [sg.Multiline(size=(60, 15), key='-LOG-', autoscroll=True, disabled=True)]
    ]

    return sg.Window('上海交通大学访客登记自动化工具', layout)


def main():
    window = create_window()

    while True:
        event, values = window.read()

        if event in (None, '退出'):
            break

        if event == '-SUBMIT-':
            # 验证输入
            if not values['-NAME-'] or not values['-ID-'] or not values['-PHONE-']:
                sg.popup_error('请填写所有字段！')
                continue

            # 确定时间选择
            time_choice = 'time1' if values['time1'] else 'time2'

            # 在新线程中运行自动化，避免界面冻结
            threading.Thread(
                target=submit_application,
                args=({
                          '-NAME-': values['-NAME-'],
                          '-ID-': values['-ID-'],
                          '-PHONE-': values['-PHONE-'],
                          '-TIME-': time_choice
                      }, window),
                daemon=True
            ).start()

    window.close()


if __name__ == "__main__":
    main()