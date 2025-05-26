# 快代理API配置
import os

import requests
import random
import time

PROXY_API_URL = "https://dps.kdlapi.com/api/getdps/?secret_id=ojbzgxwopxm5mxqx8hlh&signature=otz9s5n326r25etg1a3kqihfke&num=1&pt=1&format=json&sep=1"


class ProxyPool:
    def __init__(self):
        self.proxies = []
        self.last_update = 0
        self.username = os.getenv('KDL_USERNAME', 'your_default_username')  # 从环境变量读取
        self.password = os.getenv('KDL_PASSWORD', 'your_default_password')
        self.api_url = os.getenv('KDL_API_URL', 'your_default_URL')
        self.proxy_ip = ""

    def fetch_proxies(self):
        if self.username == 'your_default_username':
            return None
        # API接口返回的ip
        self.proxy_ip = requests.get(self.api_url).json()['data']['proxy_list']
        """从API获取新代理IP并添加认证信息"""
        proxies = {
            "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {'user': self.username, 'pwd': self.password,
                                                            'proxy': random.choice(self.proxy_ip)},
            "https": "http://%(user)s:%(pwd)s@%(proxy)s/" % {'user': self.username, 'pwd': self.password,
                                                             'proxy': random.choice(self.proxy_ip)}
        }

        return proxies


if __name__ == "__main__":
    # 测试代理功能
    pool = ProxyPool()
    pool.username = "d2531824458"
    pool.password = "n01ze8an"
    pool.api_url = "https://dps.kdlapi.com/api/getdps/?secret_id=ojbzgxwopxm5mxqx8hlh&signature=otz9s5n326r25etg1a3kqihfke&num=1&pt=1&format=json&sep=1"
    proxy = pool.fetch_proxies()

    print(proxy)
    if proxy:
        print(f"测试代理连通性...")
        target_url = "http://httpbin.org/ip"
        try:
            response = requests.get(
                target_url,
                proxies=proxy,
                timeout=10,
                verify=False
            )
            print(f"代理测试 {'成功' if response.status_code == 200 else '失败'}")
            print(response.text)
        except Exception as e:
            print(f"代理测试异常: {str(e)}")
    else:
        print("无法获取有效代理")
