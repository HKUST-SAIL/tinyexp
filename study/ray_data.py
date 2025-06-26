import os

import ray

# 1. 初始化 Ray
# 如果 Ray 已经运行，先关闭它，确保干净启动
if ray.is_initialized():
    ray.shutdown()
# 初始化 Ray，这里我们只用2个CPU核心进行本地模拟分布式
ray.init(num_cpus=2)
print("Ray initialized.")

# --- 示例 1: 从 Python 列表创建 Dataset ---
print("\n--- 示例 1: 从 Python 列表创建 Dataset ---")
# 假设我们有一些字典形式的数据
data = [
    {"id": 1, "name": "Alice", "age": 25},
    {"id": 2, "name": "Bob", "age": 30},
    {"id": 3, "name": "Charlie", "age": 35},
    {"id": 4, "name": "David", "age": 40},
    {"id": 5, "name": "Eve", "age": 28},
]

# 使用 ray.data.from_items() 从内存中的 Python 对象创建 Dataset
ds_from_items = ray.data.from_items(data)

# 查看 Dataset 的前几项 (这会触发计算)
print("Dataset created from items:")
ds_from_items.show()
print(f"Total items in dataset: {ds_from_items.count()}")

# --- 示例 2: 从 CSV 文件加载数据 ---
print("\n--- 示例 2: 从 CSV 文件加载数据 ---")
# 首先，创建一个虚拟的 CSV 文件
csv_file_path = "people_data.csv"
csv_content = """name,age,city
Alice,25,New York
Bob,30,Los Angeles
Charlie,35,Chicago
David,40,Houston
Eve,28,New York
Frank,45,Los Angeles
"""
with open(csv_file_path, "w") as f:
    f.write(csv_content)

# 使用 ray.data.read_csv() 加载 CSV 文件
ds_csv = ray.data.read_csv(csv_file_path)

print(f"Dataset loaded from '{csv_file_path}':")
ds_csv.show()
print(f"Total items in CSV dataset: {ds_csv.count()}")

# --- 示例 3: 数据转换 (map 和 filter) ---
print("\n--- 示例 3: 数据转换 (map 和 filter) ---")


# 3.1 使用 .map() 转换数据
# 目标：为每个人添加一个 'age_in_months' 列
def add_age_in_months(row):
    # row 是一个字典，代表数据集中的一行
    row["age_in_months"] = row["age"] * 12
    return row


# 应用 map 操作。注意：这里只是定义了转换，还没有实际执行
ds_mapped = ds_csv.map(add_age_in_months)

print("Dataset after map (adding 'age_in_months'):")
ds_mapped.show()  # 此时才会触发计算


# 3.2 使用 .filter() 过滤数据
# 目标：只保留年龄大于 30 的人
def is_older_than_30(row):
    return row["age"] > 30


# 应用 filter 操作
ds_filtered = ds_mapped.filter(is_older_than_30)

print("Dataset after filter (age > 30):")
ds_filtered.show()  # 此时触发计算

# --- 示例 4: 转换为 Pandas DataFrame ---
print("\n--- 示例 4: 转换为 Pandas DataFrame ---")
# 当你需要将分布式 Dataset 收集到单机内存中进行进一步分析时
# 注意：如果数据集非常大，这可能会导致内存溢出
df_result = ds_filtered.to_pandas()
print("Converted to Pandas DataFrame:")
print(df_result)

# --- 示例 5: 转换为 NumPy 数组 ---
print("\n--- 示例 5: 转换为 NumPy 数组 ---")
# 假设我们有一个纯数值的 Dataset
numeric_data = [{"x": i, "y": i * 2} for i in range(10)]
ds_numeric = ray.data.from_items(numeric_data)

# 转换为 NumPy 数组，需要指定列名
np_array = ds_numeric.to_numpy(columns=["x", "y"])
print("Converted to NumPy Array:")
print(np_array)
print(f"Shape of NumPy array: {np_array.shape}")

# --- 示例 6: 写入数据到文件 ---
print("\n--- 示例 6: 写入数据到文件 ---")
output_dir = "output_data"
# 将过滤后的数据写入到新的 CSV 文件（Ray 会自动处理分布式写入）
# 注意：Ray 会将数据写入到指定目录下的多个分片文件
ds_filtered.write_csv(output_dir)
print(f"Filtered data written to directory: {output_dir}")
# 检查写入的文件
print("Files in output directory:")
for root, dirs, files in os.walk(output_dir):
    for file in files:
        print(os.path.join(root, file))

# --- 清理 ---
# 关闭 Ray 运行时
ray.shutdown()
print("\nRay shut down.")

# 清理创建的虚拟文件和目录
os.remove(csv_file_path)
import shutil

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
print("Cleaned up dummy files.")
