import time
from random import uniform

import requests
from lxml import html
import json
import os
from urllib.parse import urljoin


def get_page(url, max_retries=3, delay=1):
    """获取网页内容，带重试机制和随机延迟"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    for attempt in range(max_retries):
        try:
            # 随机延迟（避免请求太快）
            time.sleep(uniform(delay * 0.5, delay * 1.5))  # 随机延迟 1~3 秒

            response = requests.get(url, headers=headers, timeout=10)
            content = response.content
            tree = html.fromstring(content)
            targetDiv=tree.xpath('//*[@id="vsb_content"]/table[1]/tbody')
            for column in targetDiv:
                for element_col in column:
                    for element in element_col:
                        print(element.herf)
                # print(div.text_content())
            # print(targetDiv)


            # 检查是否返回了有效内容（避免空页面）
            if not response.content:
                print(f"⚠️ 空内容: {url}")
                continue  # 重试

            return response.content

        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败（尝试 {attempt + 1}/{max_retries}）: {e}")
            if attempt == max_retries - 1:  # 最后一次尝试仍然失败
                return None

    return None  # 所有尝试均失败


if __name__ == '__main__':
    get_page("https://renshi.nwpu.edu.cn/new/szdw/jcrc.htm")
