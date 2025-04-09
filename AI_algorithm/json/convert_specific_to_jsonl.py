import json
import ijson # 確保已安裝: pip install ijson
import argparse
import os
import sys
import traceback
import random # 導入 random 模塊用於打亂

def convert_specific_format_to_jsonl(input_path, output_path, progress_interval=10000, shuffle_output=False):
    """
    将特定格式 (如样本所示) 的 JSON 文件高效转换为 JSON Lines (.jsonl)。
    内置验证: 检查 A 长度为 6, B 长度为 3。
    新增功能: 可选择在写入前打乱所有有效的 JSON 项目。

    Args:
        input_path (str): 输入 JSON 文件的路径 (假定为包含样本对象的列表)。
        output_path (str): 输出 JSONL 文件的路径。
        progress_interval (int): 每处理多少个项目打印一次进度。设为 0 可禁用。
        shuffle_output (bool): 如果为 True，则在写入前打乱所有有效的项目。
                               这会将所有有效项目加载到内存中。
    """
    print(f"开始转换特定格式文件: '{input_path}' -> '{output_path}'")
    if shuffle_output:
        print("警告: 已启用打乱输出 (--shuffle)。所有有效项目将先加载到内存中进行打乱，这可能消耗大量内存。")

    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在。", file=sys.stderr)
        return False

    # --- 变量初始化 ---
    valid_items = [] # 用于存储所有有效的项目 (如果需要打乱或统一写入)
    processed_count = 0
    skipped_count = 0
    # item_prefix: ijson 需要知道如何找到列表中的项目。对于 ['item1', 'item2', ...] 结构，'item' 是正确的。
    # 如果 JSON 结构是 {"key": ["item1", ...]}, 则前缀应为 "key.item"。
    # 假设输入是顶层列表 [...]
    item_prefix = 'item'

    try:
        # === 阶段 1: 读取、验证并将有效项目存入列表 ===
        print("阶段 1: 读取和验证输入文件中的项目...")
        with open(input_path, 'rb') as f_in:
            # 流式解析 JSON 列表中的每个项目
            parser = ijson.items(f_in, item_prefix)

            for item in parser:
                processed_count += 1
                valid_item = True
                skip_reason = ""

                # --- 内置格式验证 ---
                if not isinstance(item, dict):
                    valid_item = False
                    skip_reason = "不是 JSON 对象 (字典)"
                elif not all(k in item for k in ["A", "B", "max_score", "best_moves"]):
                    valid_item = False
                    skip_reason = "缺少必需的键 (A, B, max_score, best_moves)"
                elif not isinstance(item.get("A"), list) or not isinstance(item.get("B"), list):
                    valid_item = False
                    skip_reason = "键 'A' 或 'B' 的值不是列表"
                elif len(item["A"]) != 6 or len(item["B"]) != 3:
                    valid_item = False
                    skip_reason = f"列表长度不匹配 (A={len(item['A'])}, B={len(item['B'])}), 需要 A=6, B=3"
                elif not isinstance(item.get("best_moves"), list):
                     valid_item = False
                     skip_reason = "键 'best_moves' 的值不是列表"
                # --- 验证结束 ---

                if valid_item:
                    # 将通过验证的项目添加到列表中
                    valid_items.append(item)
                else:
                    # 如果验证失败，则跳过此项目
                    skipped_count += 1
                    # 避免过多警告，可以设置条件打印
                    if skipped_count % (progress_interval if progress_interval > 0 else 1000) == 1:
                         print(f"警告: 跳过第 {processed_count} 项。原因: {skip_reason}", file=sys.stderr)

                # 打印读取进度
                if progress_interval > 0 and processed_count % progress_interval == 0:
                    # 显示处理、当前有效和跳过的计数
                    print(f"  读取进度: {processed_count} 项已处理, {len(valid_items)} 项有效, {skipped_count} 项已跳过...", end='\r')

        print(f"\n阶段 1 完成。总共处理 {processed_count} 个原始项目。")
        print(f"其中 {len(valid_items)} 个项目通过验证，{skipped_count} 个项目被跳过。")

        # === 阶段 2: 打乱 (如果启用) ===
        if shuffle_output:
            if not valid_items:
                print("阶段 2: 没有有效的项目可供打乱。")
            else:
                print(f"阶段 2: 正在打乱 {len(valid_items)} 个有效项目... (这可能需要一些时间)")
                random.shuffle(valid_items)
                print("打乱完成。")
        else:
            print("阶段 2: 未启用打乱 (--shuffle 未设置)。将按原始顺序写入有效项目。")

        # === 阶段 3: 将列表中的项目写入输出文件 ===
        print(f"阶段 3: 正在将 {len(valid_items)} 个项目写入输出文件 '{output_path}'...")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"错误: 无法创建输出目录 '{output_dir}'. {e}", file=sys.stderr)
                # 如果无法创建目录，后续写入会失败，这里可以直接返回 False
                return False

        final_written_count = 0
        serialization_skipped_count = 0
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, item in enumerate(valid_items):
                try:
                    # ensure_ascii=False 保证中文等字符正确输出
                    json_string = json.dumps(item, ensure_ascii=False)
                    f_out.write(json_string + '\n')
                    final_written_count += 1
                except TypeError as e:
                    # 如果序列化失败（理论上不太可能，因为是从有效 JSON 读入的）
                    serialization_skipped_count += 1
                    print(f"\n警告: 无法序列化第 {i+1} 个有效项目 (在写入阶段)。跳过。错误: {e}", file=sys.stderr)
                    continue # 继续处理下一个项目

                # 打印写入进度
                if progress_interval > 0 and (i + 1) % progress_interval == 0:
                    print(f"  写入进度: {i + 1}/{len(valid_items)}...", end='\r')

        print(f"\n阶段 3 完成。成功写入 {final_written_count} 个项目。")
        if serialization_skipped_count > 0:
             print(f"在写入阶段因序列化错误跳过了 {serialization_skipped_count} 个项目。")


        # === 转换完成 ===
        total_skipped = skipped_count + serialization_skipped_count
        print(f"\n转换过程结束。")
        print(f"总共处理原始项目数: {processed_count}")
        print(f"符合格式且成功写入的项目数: {final_written_count}")
        print(f"因格式/验证/序列化错误而跳过的项目总数: {total_skipped}")
        print(f"输出文件已保存至: '{output_path}'")
        return True

    except ijson.JSONError as e:
        print(f"\n错误: 输入 JSON 文件 '{input_path}' 格式无效或在阶段 1 解析中途失败。", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        print(f"处理到大约第 {processed_count} 个项目时出错。", file=sys.stderr)
        return False
    except IOError as e:
        print(f"\n错误: 读写文件时发生错误。", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        return False
    except MemoryError:
        print(f"\n错误: 内存不足！", file=sys.stderr)
        print(f"在处理了 {processed_count} 个项目，收集了 {len(valid_items)} 个有效项目后发生内存错误。", file=sys.stderr)
        print(f"这通常发生在尝试将过多有效数据加载到内存中以进行打乱时。", file=sys.stderr)
        print(f"如果输入文件包含大量有效项目，请考虑在不使用 --shuffle 选项的情况下运行，", file=sys.stderr)
        print(f"或者使用能处理大文件打乱的外部工具。", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n发生意外错误:", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将特定格式 (A[6], B[3], ...) 的 JSON 文件高效转换为 JSON Lines (.jsonl)，并可选择打乱输出。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 显示默认值
    )
    parser.add_argument(
        "input_file",
        help="输入的 JSON 文件路径 (包含对象列表)。"
    )
    parser.add_argument(
        "output_file",
        help="输出的 JSONL 文件路径。"
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=10000,
        help="每处理 N 个项目显示一次进度。设置为 0 可禁用。"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true", # 当出现 --shuffle 时，其值为 True
        help="在写入输出文件前，打乱所有有效的 JSON 项目。注意：这会将所有有效项目加载到内存中。"
    )

    args = parser.parse_args()

    # 执行转换，传入 shuffle 参数
    success = convert_specific_format_to_jsonl(
        args.input_file,
        args.output_file,
        args.progress,
        args.shuffle # 将命令行参数传递给函数
    )

    # 根据转换结果退出，0 表示成功，非 0 表示失败
    sys.exit(0 if success else 1)