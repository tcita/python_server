import json
import multiprocessing as mp
from tqdm import tqdm
import os

from AI_algorithm.brute_force import recursive_StrategyAndScore
from AI_algorithm.tool.tool import deal_cards_tool


def process_batch(batch_size, batch_id, output_file):
    dataset = []
    whole = [0, 1, 2]

    for _ in range(batch_size):
        A, B = deal_cards_tool()  # 初始A, B   A, B 都是 list<int>
        max_score, best_moves = recursive_StrategyAndScore(A, B)

        if len(best_moves) != 3:
            continue
        # 处理best_moves不足3个的情况
        current_rows = set(move[0] for move in best_moves)
        remaining = [i for i in whole if i not in current_rows]

        for i in remaining:
            if len(best_moves) < 3:
                best_moves.append([i, 0])

        dataset.append({"A": A, "B": B, "max_score": max_score, "best_moves": best_moves})

    # 每个批次单独写入文件
    with open(f"{output_file}_{batch_id}.json", "w") as f:
        json.dump(dataset, f)

    return batch_id


def merge_batch_files(num_batches, output_file, batch_file_prefix):
    merged_data = []

    for i in range(num_batches):
        batch_file = f"{batch_file_prefix}_{i}.json"
        if os.path.exists(batch_file):
            with open(batch_file, "r") as f:
                batch_data = json.load(f)
                merged_data.extend(batch_data)
            # 删除临时文件
            os.remove(batch_file)

    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=4)


def generate_training_data(num_samples=50000):
    # 确保输出目录存在
    os.makedirs("json", exist_ok=True)

    # 计算批次数和每批样本数
    num_cores = mp.cpu_count()
    batch_size = min(10000, num_samples // num_cores)  # 每批不超过10000个样本
    num_batches = (num_samples + batch_size - 1) // batch_size  # 向上取整

    print(f"使用{num_cores}个核心，分{num_batches}批处理，每批{batch_size}个样本")

    # 临时文件前缀
    temp_file_prefix = "json/data_raw_temp"

    # 创建进程池
    pool = mp.Pool(processes=num_cores)

    # 提交任务
    results = []
    for i in range(num_batches):
        # 最后一批可能不足batch_size
        actual_batch_size = min(batch_size, num_samples - i * batch_size)
        if actual_batch_size <= 0:
            break
        result = pool.apply_async(process_batch, (actual_batch_size, i, temp_file_prefix))
        results.append(result)

    # 显示进度条并等待所有任务完成
    with tqdm(total=num_batches) as pbar:
        for result in results:
            result.get()  # 等待任务完成
            pbar.update(1)

    pool.close()
    pool.join()

    # 合并所有批次文件
    print("合并批次文件...")
    merge_batch_files(num_batches, "json/data_Trans_skip.json", temp_file_prefix)
    print("数据集生成完成！")


if __name__ == "__main__":
    generate_training_data(num_samples=5000000)