import json
import ijson # 确保已安装: pip install ijson
import argparse
import os
import sys
import traceback

def convert_specific_format_to_jsonl(input_path, output_path, progress_interval=10000):
    """
    将特定格式 (如样本所示) 的 JSON 文件高效转换为 JSON Lines (.jsonl)。
    内置验证: 检查 A 长度为 6, B 长度为 3。

    Args:
        input_path (str): 输入 JSON 文件的路径 (假定为包含样本对象的列表)。
        output_path (str): 输出 JSONL 文件的路径。
        progress_interval (int): 每处理多少个项目打印一次进度。设为 0 可禁用。
    """
    print(f"开始转换特定格式文件: '{input_path}' -> '{output_path}'")

    # 检查输入文件
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 '{input_path}' 不存在。", file=sys.stderr)
        return False

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"错误: 无法创建输出目录 '{output_dir}'. {e}", file=sys.stderr)
            return False

    processed_count = 0
    kept_count = 0
    skipped_count = 0
    # 硬编码项目前缀，因为我们假设输入是顶层列表 [...]
    item_prefix = 'item'

    try:
        # 以二进制模式打开输入文件供 ijson 使用
        # 以 UTF-8 文本模式打开输出文件
        with open(input_path, 'rb') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            print(f"使用内置前缀 '{item_prefix}' 解析项目...")
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
                # 检查必需的键
                elif not all(k in item for k in ["A", "B", "max_score", "best_moves"]):
                    valid_item = False
                    skip_reason = "缺少必需的键 (A, B, max_score, best_moves)"
                # 检查 A 和 B 是否为列表
                elif not isinstance(item.get("A"), list) or not isinstance(item.get("B"), list):
                    valid_item = False
                    skip_reason = "键 'A' 或 'B' 的值不是列表"
                # **关键验证：检查 A 和 B 的长度是否符合 6x3 要求**
                elif len(item["A"]) != 6 or len(item["B"]) != 3:
                    valid_item = False
                    skip_reason = f"列表长度不匹配 (A={len(item['A'])}, B={len(item['B'])}), 需要 A=6, B=3"
                # 可以添加更多验证，例如检查 best_moves 的结构
                elif not isinstance(item.get("best_moves"), list):
                     valid_item = False
                     skip_reason = "键 'best_moves' 的值不是列表"
                # --- 验证结束 ---

                # 如果验证失败，则跳过此项目
                if not valid_item:
                    skipped_count += 1
                    # 避免过多警告，可以设置条件打印
                    if skipped_count % (progress_interval if progress_interval > 0 else 1000) == 1:
                         print(f"警告: 跳过第 {processed_count} 项。原因: {skip_reason}", file=sys.stderr)
                    continue

                # 如果项目有效，则将其转换为 JSON 字符串并写入文件
                try:
                    # ensure_ascii=False 保证中文等字符正确输出
                    json_string = json.dumps(item, ensure_ascii=False)
                    f_out.write(json_string + '\n')
                    kept_count += 1
                except TypeError as e:
                    # 如果序列化失败（不太可能，除非数据内部有问题）
                    skipped_count += 1
                    print(f"警告: 无法序列化第 {processed_count} 项 (已通过验证)。跳过。错误: {e}", file=sys.stderr)
                    continue

                # 打印进度
                if progress_interval > 0 and processed_count % progress_interval == 0:
                    # 显示处理、保留和跳过的计数
                    print(f"进度: 已处理 {processed_count}, 已保留 {kept_count}, 已跳过 {skipped_count}...")

        # 转换完成后的最终报告
        print(f"\n转换完成。")
        print(f"总共处理项目数: {processed_count}")
        print(f"符合格式并保留的项目数: {kept_count}")
        print(f"因格式不符或错误而跳过的项目数: {skipped_count}")
        print(f"输出文件已保存至: '{output_path}'")
        return True

    except ijson.JSONError as e:
        print(f"\n错误: 输入 JSON 文件 '{input_path}' 格式无效或解析中途失败。", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        print(f"处理到大约第 {processed_count} 个项目时出错。", file=sys.stderr)
        return False
    except IOError as e:
        print(f"\n错误: 读写文件时发生错误。", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n发生意外错误:", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将特定格式 (A[6], B[3], ...) 的 JSON 文件转换为 JSON Lines (.jsonl)。",
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

    args = parser.parse_args()

    # 执行转换
    success = convert_specific_format_to_jsonl(
        args.input_file,
        args.output_file,
        args.progress
    )

    # 根据转换结果退出，0 表示成功，非 0 表示失败
    sys.exit(0 if success else 1)