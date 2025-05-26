import time
from random import uniform

import requests
from lxml import html
import json
import os
from urllib.parse import urljoin


def get_page(url="https://jsj.nwpu.edu.cn/snew/szdw.htm", max_retries=3, delay=1):
    """获取网页内容，带重试机制和随机延迟"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    # 随机延迟（避免请求太快）
    time.sleep(uniform(delay * 0.5, delay * 1.5))  # 随机延迟 1~3 秒

    response = requests.get(url, headers=headers, timeout=10)
    content = response.content
    tree = html.fromstring(content)
    print(content)

if __name__ == '__main__':
    get_page()