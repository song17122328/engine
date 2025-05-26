import requests
from datetime import datetime
import time

# 配置信息（需替换成你的信息）

REFERER_URL = "https://qiandao.sjtu.edu.cn/visitor/submit.php"  # 替换成登记页面URL
USER_AGENT = "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36 MicroMessenger/8.0.25.2140(0x28001937)"

# 你的个人信息
FORM_DATA = {
    "campus": "闵行校区",  # 可能服务器自动填充，留空或填写默认值
    "time": 2,
    "xm": "胡伊仑",  # 替换成你的真实姓名 海南省  小明建模
    "zjhm": "330103199707250428",  # 替换成你的身份证号/学号 小明剑魔
    "phone": "17821696076",  # 替换成你的手机号
}


def submit_application():
    cookies = {
        # 闵行校区的cookies
        "mh": {"ik": "ffbec42ee7",
               "10.119.6.139:80": "22632.59273.21071.0000",
               "VISITOR": "3c8h88lukkvob21c6hg8phpsie"},
        # 徐汇校区的cookies
        "xh": {"ik": "ff19d84eaf",
               "10.119.6.139: 80": "22632.59273.21071.0000",
               "VISITOR": "tea0m901d82rn50qudihqe2iha"
               }
    }
    # 2. 构造请求头（防止被拦截）
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36 MicroMessenger/8.0.25.2140(0x28001937)",
        "Referer": "https://qiandao.sjtu.edu.cn/visitor/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Origin": "https://qiandao.sjtu.edu.cn",  # 替换成学校域名

        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Host": "qiandao.sjtu.edu.cn",  # 替换成学校域名
        "content-type": "application/x-www-form-urlencoded",

    }

    # 3. 发送 POST 请求
    response = requests.post(
        REFERER_URL,
        data=FORM_DATA,
        headers=headers,
        cookies=cookies["mh"],
        timeout=10,
    )

    # 4. 检查是否成功
    if response.status_code == 200:
        print(f"[{datetime.now()}] 提交成功！响应内容:", response.text)
    else:
        print(f"[{datetime.now()}] 提交失败！HTTP状态码:", response.status_code)


# 定时任务（每天 8:00 和 13:00 提交）
if __name__ == "__main__":
    submit_application()
