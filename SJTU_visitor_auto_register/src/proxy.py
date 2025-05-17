# 快代理API配置
import requests
import random
import time
PROXY_API_URL = "https://dps.kdlapi.com/api/getdps/?secret_id=ojbzgxwopxm5mxqx8hlh&signature=oni6vnvylyolucfd72tc37xtu0&num=1&pt=1&format=json&sep=1"
TARGET_URL = "https://qiandao.sjtu.edu.cn/visitor/submit.php"  # 你的目标网站


class ProxyPool:
    def __init__(self):
        self.proxies = []
        self.last_update = 0

    def fetch_proxies(self):
        """从快代理API获取新IP"""
        # try:

        resp = requests.get(PROXY_API_URL, timeout=10,verify=False).json()
        if resp['code'] == 0:
            self.proxies = [
                f"http://{proxy}"
                for proxy in resp['data']['proxy_list']
            ]
            self.last_update = time.time()
            print(f"成功获取 {len(self.proxies)} 个代理IP")
        # except Exception as e:
        #     print(f"获取代理失败: {e}")

    def get_random_proxy(self):
        """获取随机可用代理"""
        print("获取代理中...")
        if not self.proxies or time.time() - self.last_update > 300:  # 5分钟更新一次
            self.fetch_proxies()
        return random.choice(self.proxies) if self.proxies else None

    def validate_proxy(self, proxy):
        """验证代理是否有效"""
        try:
            test_url = "http://httpbin.org/ip"  # 测试网站
            resp = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=5)
            return resp.status_code == 200
        except:
            return False


