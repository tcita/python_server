import json
import multiprocessing
import os
import random
import time
from collections import defaultdict

from AI_algorithm.brute_force import recursive_StrategyAndScore
from AI_algorithm.tool.tool import deal_cards_tool


def generate_data_worker(process_id, bin_ranges, samples_per_bin, return_dict, shared_bin_counts, lock):
    """
    Worker process function, modified to update shared counts.
    Args:
        process_id: Unique ID for the process.
        bin_ranges: List of tuples defining score ranges for bins.
        samples_per_bin: Target number of samples per bin for *this* worker.
        return_dict: Manager.dict to store final results (filled at the end).
        shared_bin_counts: Manager.dict to store real-time counts per bin across all workers.
        lock: Manager.Lock for safe concurrent updates to shared_bin_counts.
    """
    local_bins = defaultdict(list)
    filled_bins_count = 0 # 本地计数器：记录本进程完成了几个 bin 的填充目标
    target_bins_to_fill = len(bin_ranges)
    local_samples_generated = 0  # 本进程尝试生成的数据条数

    print(f"进程 {process_id} (PID: {os.getpid()}) 开始运行, 每个桶目标 {samples_per_bin} 条数据。")

    # 循环条件改为检查本进程是否完成了所有桶的配额
    while filled_bins_count < target_bins_to_fill:
        try:
            A, B = deal_cards_tool()
            max_score, best_moves = recursive_StrategyAndScore(A, B)
        except Exception as e:
            print(f"进程 {process_id} 在生成数据时遇到错误: {e}")
            time.sleep(0.1) # 避免错误导致空转
            continue

        local_samples_generated += 1

        # 判断落在哪个 bin
        for i, (low, high) in enumerate(bin_ranges):
            # 检查分数是否在区间内，且该 bin 在本进程内尚未填满
            if low <= max_score < high and len(local_bins[i]) < samples_per_bin:
                # 补足 best_moves (保持原有逻辑)
                rows = set(move[0] for move in best_moves)
                # 注意：这里的逻辑是确保至少有3个move，不足补 [j, 0]
                # 如果 recursive_StrategyAndScore 可能返回超过3个move，这个逻辑可能需要调整
                # 但基于原代码，我们保持这个逻辑
                valid_rows = {0, 1, 2}
                current_rows = set(move[0] for move in best_moves)
                moves_to_add = []
                for j in valid_rows:
                   if j not in current_rows:
                       moves_to_add.append([j, 0])

                # 添加补足的 moves，确保总数不超过 3 (如果原始 moves 加上补充超过3)
                # 或者按原代码逻辑，只要原始不足3就补充，可能超过3？假设按原代码逻辑直接添加
                best_moves.extend(moves_to_add)
                # 如果需要严格限制为3个，需要截断: best_moves = (best_moves + moves_to_add)[:3]

                # 添加数据到本地 bin
                local_bins[i].append({"A": A, "B": B, "max_score": max_score, "best_moves": best_moves})

                # --- 新增：更新共享计数器 ---
                with lock:
                    shared_bin_counts[i] += 1
                # --- 结束新增 ---

                # 检查本进程是否填满了这个 bin
                if len(local_bins[i]) == samples_per_bin:
                    filled_bins_count += 1
                    # print(f"进程 {process_id} 完成了桶 {i} 的本地配额。") # 可以取消注释用于调试

                break  # 一旦找到并处理了桶就跳出内层循环

        # 移除原有的基于 filled_bins 的 progress 更新 (现在由主进程负责全局进度)
        # if local_count % 100 == 0:
        #     progress.value = sum(len(local_bins[i]) for i in filled_bins) / total_samples * 100

        # 可以保留或调整这个内部进度打印
        if local_samples_generated % 5000 == 0: # 调整打印频率
            print(f"进程 {process_id} 已尝试生成 {local_samples_generated} 条数据 (已完成 {filled_bins_count}/{target_bins_to_fill} 个本地桶配额)")

    print(f"进程 {process_id} (PID: {os.getpid()}) 已完成所有本地桶配额，将结果写入 return_dict。")
    # 将 defaultdict 转换为普通 dict 存入 Manager.dict
    return_dict[process_id] = dict(local_bins)


# 修改后的 parallel_generate 函数
def parallel_generate(samples_per_bin=10000, num_workers=8):
    bin_ranges = [(20 + i * 10, 20 + (i + 1) * 10) for i in range(8)]  # 8个桶
    num_bins = len(bin_ranges)
    # 注意：这里的 samples_per_bin 是每个 worker 需要为每个 bin 生成的数量
    # 总目标是 samples_per_bin * num_workers
    target_samples_per_bin_total = samples_per_bin * num_workers

    manager = multiprocessing.Manager()
    return_dict = manager.dict() # 用于收集每个进程最终的数据
    # --- 新增：共享计数器和锁 ---
    shared_bin_counts = manager.dict({i: 0 for i in range(num_bins)})
    lock = manager.Lock()
    # --- 结束新增 ---
    # 移除不再需要的 progress Value 对象
    # progress = multiprocessing.Value('d', 0.0)
    jobs = []

    print(f"启动 {num_workers} 个工作进程，每个进程为每个桶生成 {samples_per_bin} 条数据...")
    # 启动进程
    for i in range(num_workers):
        # --- 修改：传递 shared_bin_counts 和 lock 给 worker ---
        p = multiprocessing.Process(target=generate_data_worker,
                                    args=(i, bin_ranges, samples_per_bin, return_dict, shared_bin_counts, lock))
        jobs.append(p)
        p.start()

    # 主进程负责输出进度
    monitoring_start_time = time.time()  # 记录开始时间
    last_print_time = time.time()

    while any(p.is_alive() for p in jobs):
        current_time = time.time()
        # --- 修改：从 shared_bin_counts 读取进度 ---
        if current_time - last_print_time >= 10:  # 每隔 10 秒输出一次
            print(f"\n--- 进度更新 (已运行 {current_time - monitoring_start_time:.2f} 秒) ---")
            total_generated_so_far = 0
            # 读取共享计数器的当前值
            with lock: # 加锁读取确保一致性
                current_counts = dict(shared_bin_counts)

            for i in range(num_bins):
                count = current_counts.get(i, 0)
                total_generated_so_far += count
                print(f"桶 {i}（{bin_ranges[i][0]} ~ {bin_ranges[i][1]}）: {count}/{target_samples_per_bin_total} 条数据")

            total_target_samples = target_samples_per_bin_total * num_bins
            percentage = (total_generated_so_far / total_target_samples) * 100 if total_target_samples > 0 else 0
            print(f"总计: {total_generated_so_far} / {total_target_samples} ({percentage:.2f}%)")
            print("--- 结束进度更新 ---\n")

            last_print_time = current_time # 重置计时

        time.sleep(1)  # 每秒检查一次

    # 等待所有进程结束
    print("\n等待所有进程完成...")
    for p in jobs:
        p.join()

    print("\n所有进程完成，开始合并数据...")

    # 合并所有进程的数据 (这部分逻辑保持不变)
    merged_bins = defaultdict(list)
    final_data_count = 0
    if len(return_dict.keys()) != num_workers:
         print(f"警告: 只有 {len(return_dict.keys())}/{num_workers} 个进程成功返回了数据。")

    for worker_id in return_dict.keys():
        worker_data = return_dict[worker_id]
        if not isinstance(worker_data, dict):
            print(f"警告: 进程 {worker_id} 返回的数据类型不是字典: {type(worker_data)}")
            continue
        for bin_index in range(num_bins):
             # 确保 worker_data 中确实有这个 bin_index 的数据
             if bin_index in worker_data:
                 data_to_add = worker_data[bin_index]
                 if isinstance(data_to_add, list):
                    merged_bins[bin_index].extend(data_to_add)
                 else:
                    print(f"警告: 进程 {worker_id} 的桶 {bin_index} 数据类型不是列表: {type(data_to_add)}")


    # 打乱数据 (这部分逻辑保持不变)
    all_data = []
    for bin_index in range(num_bins):
        bin_data = merged_bins[bin_index]
        print(f"桶 {bin_index}: 合并后得到 {len(bin_data)} 条数据")
        final_data_count += len(bin_data)
        all_data.extend(bin_data)

    print(f"合并后总数据量: {final_data_count}")
    # 验证计数是否大致吻合
    final_shared_total = sum(shared_bin_counts.values())
    print(f"共享计数器最终总和: {final_shared_total}")
    if final_data_count != final_shared_total:
        print(f"警告: 合并后的数据量 ({final_data_count}) 与共享计数器总和 ({final_shared_total}) 不符，请检查逻辑。")


    print("打乱数据...")
    random.shuffle(all_data)

    # 保存到文件 (这部分逻辑保持不变)
    output_filename = "json/data_raw.json"  # 修改文件名以示区别
    os.makedirs(os.path.dirname(output_filename), exist_ok=True) # 确保目录存在
    print(f"保存数据到 {output_filename}...")
    try:
        with open(output_filename, "w") as f:
            json.dump(all_data, f, indent=4)
        print(f"✅ 并行采样完成，总数据量：{len(all_data)}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


if __name__ == '__main__':
    # 确保 'json' 目录存在
    if not os.path.exists('json'):
        os.makedirs('json')

    # 使用原始参数运行
    parallel_generate(samples_per_bin=10000, num_workers=8)
    # 或者使用较小的参数进行快速测试
    # parallel_generate(samples_per_bin=100, num_workers=4)