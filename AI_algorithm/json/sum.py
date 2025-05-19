import json
import sys

# 假设你的 JSON 数据存储在一个文件中，或者你可以直接将其作为字符串加载
# 如果数据在文件中，请确保文件路径正确
file_path = "data_raw2.json"

# 加载 JSON 数据
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 确保数据是一个列表（JSON 数组）
if isinstance(data, list):
    # 统计样本数
    sample_count = len(data)
    print(f"数据集的样本总数为: {sample_count}")

    # 计算 JSON 数据在内存中的大小
    total_memory_size_bytes = 0
    total_object_memory_size = 0

    for sample in data:
        # 方法 1: 将每个样本重新序列化为 JSON 字符串并计算字节大小
        json_string = json.dumps(sample)
        memory_size_bytes = len(json_string.encode('utf-8'))  # 计算 UTF-8 编码后的字节大小
        total_memory_size_bytes += memory_size_bytes

        # 方法 2: 使用 sys.getsizeof 估算 Python 对象的内存占用
        object_memory_size = sys.getsizeof(sample)
        total_object_memory_size += object_memory_size

    # 总结内存占用
    total_memory_size_kb = total_memory_size_bytes / 1024  # 转换为 KB
    total_memory_size_mb = total_memory_size_kb / 1024  # 转换为 MB

    print(
        f"JSON 数据在内存中的总大小（UTF-8 编码）: {total_memory_size_bytes} 字节 ({total_memory_size_kb:.2f} KB, {total_memory_size_mb:.2f} MB)")
    print(f"Python 对象本身的总内存占用: {total_object_memory_size} 字节")
else:
    print("数据集不是有效的 JSON 数组格式！")