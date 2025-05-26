import time
from random import uniform

import requests
from lxml import html
import json
import os
from urllib.parse import urljoin

base_url = "https://renshi.nwpu.edu.cn/new/szdw/jcrc.htm"  # 必须包含协议和域名

# 创建保存目录
os.makedirs("downloaded_pages", exist_ok=True)


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
            response.raise_for_status()  # 检查 HTTP 状态码（200-299）

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


def parse_list_page(content, path='/html/body/div[5]/div/div[2]/div[2]/ul[1]'):
    path += '/li/a[@href and @title]'
    tree = html.fromstring(content)
    links = tree.xpath(path)  # 更通用的XPath
    return [
        {
            'title': a.xpath('@title')[0],
            'relative_url': a.xpath('@href')[0],
            'absolute_url': urljoin(base_url, a.xpath('@href')[0])
        }
        for a in links
    ]


def parse_detail_page(content):
    """解析详情页内容，处理可能的异常"""
    if content is None:
        print(f"⚠️ 页面内容为空: ")
        return None

    try:
        tree = html.fromstring(content)
        target_div = tree.xpath('//*[@id="vsb_content"]')

        if not target_div:
            print(f"⚠️ 未找到目标内容: ")
            return None

        all_text = target_div[0].xpath("string(.)")
        cleaned_text = ' '.join(all_text.strip().split())

        print("提取结果：")
        print(cleaned_text)
        return cleaned_text if cleaned_text else None

    except Exception as e:
        print(f"❌ 解析失败: - {e}")
        return None


def save_data(data, filename='教授'):
    filename += '.json'
    # 保存为JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(start_url, filename):
    # 1. 获取初始页面
    list_page = get_page(start_url)
    if not list_page:
        return

    # 2. 解析链接列表

    items = parse_list_page(list_page)

    # 3. 获取每个链接的内容

    for item in items:
        print(f"正在处理: {item['title']}")
        page_content = get_page(item['absolute_url'])
        item['html_content'] = parse_detail_page(page_content)

    # 4.把得到的完整items保存到JSON文件
    # print(items)
    save_data(items, filename)

    print("完成！结果已保存到output.json和downloaded_pages目录")


if __name__ == "__main__":
    # 同济大学
    # 2.1. 教授
    pro_url = "https://cs.tongji.edu.cn/szdw/jsml_azc_/js_yjy_.htm"
    # 副教授
    path_pro_ = "https://cs.tongji.edu.cn/szdw/jsml_azc_/fjs_fyjy_.htm"
    main(path_pro_, '副教授')
    # 助理教授
    path_yupin = "https://cs.tongji.edu.cn/szdw/jsml_azc_/ypzljs.htm"
    main(path_yupin, '助理教授')
    # 讲师
    path_jiangshi = "https://cs.tongji.edu.cn/szdw/jsml_azc_/js1.htm"
    main(path_jiangshi, '讲师')
