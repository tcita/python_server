import json
import os # 导入 os 模块来检查文件是否存在

def deduplicate_dataset(dataset):
    """
    根据 'A', 'B', 和 'best_moves' 字段对数据集进行去重。
    (此函数与之前的版本相同)

    Args:
        dataset: 一个包含字典的列表，每个字典代表一个数据样本。

    Returns:
        一个列表，包含去重后的数据样本。
    """
    unique_samples = []
    seen_keys = set() # 使用集合来存储已经见过的组合，以提高查找效率

    for sample in dataset:
        # 检查必要的键是否存在
        if not all(k in sample for k in ["A", "B", "best_moves"]):
            print(f"警告: 样本 {sample} 缺少必要的键 ('A', 'B', 或 'best_moves')，将被跳过。")
            continue

        # 将列表 'A' 和 'B' 转换为元组，因为列表是不可哈希的
        # 添加类型检查以增加稳健性
        try:
            if not isinstance(sample["A"], list) or not isinstance(sample["B"], list):
                 print(f"警告: 样本 {sample} 的 'A' 或 'B' 不是列表，将被跳过。")
                 continue
            key_a = tuple(sample["A"])
            key_b = tuple(sample["B"])

            # 将 'best_moves'（列表的列表）转换为元组的元组
            if not isinstance(sample["best_moves"], list):
                 print(f"警告: 样本 {sample} 的 'best_moves' 不是列表，将被跳过。")
                 continue
            # 确保 best_moves 内部元素是列表或元组并且可以转换为元组
            key_best_moves = tuple(tuple(move) for move in sample["best_moves"] if isinstance(move, (list, tuple)))
            # 如果转换后长度不匹配（说明有非列表/元组元素），可能需要发出警告或跳过
            if len(key_best_moves) != len(sample["best_moves"]):
                 print(f"警告: 样本 {sample} 的 'best_moves' 包含非列表/元组元素，可能影响去重准确性。")

        except TypeError as e:
             # 捕获在转换为元组时可能发生的错误 (例如，如果列表包含不可哈希的类型)
             print(f"警告: 处理样本 {sample} 时创建唯一键出错 ({e})，将被跳过。")
             continue
        except KeyError as e:
             # 应该不会发生，因为前面有检查，但以防万一
             print(f"警告: 样本 {sample} 缺少键 {e}，将被跳过。")
             continue


        # 创建一个组合键，用于唯一标识一个样本（基于 A, B, best_moves）
        composite_key = (key_a, key_b, key_best_moves)

        # 检查这个组合键是否已经见过
        if composite_key not in seen_keys:
            # 如果没见过，将其添加到 seen_keys 集合中
            seen_keys.add(composite_key)
            # 并将原始样本添加到 unique_samples 列表中
            unique_samples.append(sample)
        # else: # 如果 composite_key 已经在 seen_keys 中，则忽略这个样本

    return unique_samples

# --- 主脚本执行部分 ---

input_filename = "data_raw.json"
output_filename = "data_unique.json" # 可以将结果保存到这个新文件

# 1. 检查输入文件是否存在
if not os.path.exists(input_filename):
    print(f"错误: 输入文件 '{input_filename}' 不存在。请确保文件路径正确。")
else:
    print(f"找到输入文件: '{input_filename}'")
    your_dataset = []
    # 2. 读取 JSON 文件
    try:
        # 使用 'utf-8' 编码打开文件，更具兼容性
        with open(input_filename, 'r', encoding='utf-8') as f:
            your_dataset = json.load(f)
        print(f"成功从 '{input_filename}' 加载了 {len(your_dataset)} 条记录。")

        # 确保加载的数据是列表格式
        if not isinstance(your_dataset, list):
            print(f"错误: '{input_filename}' 的内容不是一个有效的 JSON 列表。脚本需要列表格式的数据。")
        else:
            # 3. 执行去重
            print("开始去重处理...")
            unique_dataset = deduplicate_dataset(your_dataset)
            print("去重处理完成。")

            # 4. 打印结果统计
            print(f"\n--- 去重结果 ---")
            print(f"原始记录数: {len(your_dataset)}")
            print(f"去重后记录数: {len(unique_dataset)}")
            removed_count = len(your_dataset) - len(unique_dataset)
            print(f"移除的重复记录数: {removed_count}")

            # 5. 将去重后的结果写入新的 JSON 文件
            write_output = True # 设置为 True 来保存文件, 设置为 False 则不保存
            if write_output:
                print(f"\n正在将去重后的数据写入到 '{output_filename}'...")
                try:
                    with open(output_filename, 'w', encoding='utf-8') as f:
                        # indent=4 使输出的 JSON 文件格式更易读
                        # ensure_ascii=False 确保非 ASCII 字符（如中文）正确写入
                        json.dump(unique_dataset, f, indent=4, ensure_ascii=False)
                    print(f"成功将 {len(unique_dataset)} 条唯一记录写入 '{output_filename}'。")
                except IOError as e:
                    print(f"错误: 无法写入文件 '{output_filename}': {e}")
                except Exception as e:
                    print(f"写入文件时发生未知错误: {e}")
            else:
                 print(f"\n未将结果写入文件（write_output 设置为 False）。")
                 # 如果不想写入文件，可以取消注释下面这行来打印部分结果到控制台
                 # print("\n去重后的部分数据示例:")
                 # print(json.dumps(unique_dataset[:5], indent=4, ensure_ascii=False)) # 打印前5条

    except json.JSONDecodeError as e:
        print(f"错误: 解析 JSON 文件 '{input_filename}' 失败。请检查文件格式是否正确。错误信息: {e}")
    except IOError as e:
        print(f"错误: 读取文件 '{input_filename}' 时发生 IO 错误: {e}")
    except Exception as e:
        # 捕获其他可能的意外错误
        print(f"处理文件时发生未知错误: {e}")