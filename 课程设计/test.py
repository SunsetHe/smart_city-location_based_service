import pandas as pd
import os

# 读取 CSV 文件
csv_file = "taxi_data_with_headers.csv"  # 替换为你的 CSV 文件路径
data = pd.read_csv(csv_file)

# 创建 traj 文件夹
output_folder = "traj"
os.makedirs(output_folder, exist_ok=True)

# 按 taxi_id 分组
grouped = data.groupby("ID")  # 替换为你的第一列列名

# 遍历每个分组
for taxi_id, group in grouped:
    # 按时间列排序
    group = group.sort_values(by="Time")  # 替换为你的时间列列名

    # 输出到 traj 文件夹
    output_file = os.path.join(output_folder, f"traj_{taxi_id}.csv")
    group.to_csv(output_file, index=False, encoding="utf-8")
    print(f"{taxi_id} has been saved")

print(f"数据已按 taxi_id 分组并保存到 {output_folder} 文件夹中。")
