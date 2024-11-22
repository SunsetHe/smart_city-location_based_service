import requests
import pandas as pd
from bs4 import BeautifulSoup

# 目标数据的 URL
url = "https://www-users.cse.umn.edu/~tianhe/BIGDATA/Feeder/TaxiData/TaxiData"

# 获取网页内容
response = requests.get(url)
if response.status_code == 200:
    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 找到 <pre> 标签中的数据
    pre_tag = soup.find("pre")
    if pre_tag:
        # 获取数据并去掉多余的空白
        raw_data = pre_tag.get_text(strip=True)

        # 分割每一行数据（按换行符或空格处理）
        lines = raw_data.splitlines()

        # 将数据拆分成列表（按逗号分隔）
        rows = [line.split(',') for line in lines]

        # 转为 DataFrame
        df = pd.DataFrame(rows, columns=["ID", "Time", "Longitude", "Latitude", "Status"])

        # 保存为 CSV 文件
        df.to_csv("taxi_data.csv", index=False, encoding="utf-8")
        print("数据已成功保存为 taxi_data.csv")
    else:
        print("未找到 <pre> 标签，可能网页结构有变化")
else:
    print("无法访问数据，状态码：", response.status_code)
