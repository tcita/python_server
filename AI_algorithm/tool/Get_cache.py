import random
import os
import pickle
import itertools
from tqdm import tqdm

from AI_algorithm.tool.tool import init_deck_tool, simulate_insertion_tool


def generate_Aandx():
    deck = init_deck_tool()
    random.shuffle(deck)
    a_unique = set()
    A = []
    while len(A) < 6:
        card = deck.pop()
        if card not in a_unique:
            a_unique.add(card)
            A.append(card)

    x = deck.pop()
    return A, x


def generate_all_Aandx_combinations(a_length=2):
    """
    生成所有可能的 A 和 x 组合
    A 是 a_length 个不同的数字（1-13），x 是 1-13 中的一个数字

    参数:
        a_length: A 的长度，默认为 2

    返回:
        generator: 生成器，每次返回一个 (A, x) 元组
    """
    # 生成所有可能的 A 组合（从 1-13 中选择 a_length 个不同的数字）
    for A_perm in itertools.permutations(range(1, 14), a_length):
        # 对于每个 A 组合，x 可以是 1-13 中的任何一个数字
        for x in range(1, 14):
            yield list(A_perm), x


# 全局缓存字典
_simulation_cache = {}


def get_cache_key(A, x, pos):
    """生成基于A, x和pos的唯一缓存键"""
    # 将A转换为元组，因为列表不可哈希
    return (tuple(A), x, pos)


def build_simulation_cache(a_length=2, save_path='../cache/simulation_cache.pkl'):
    """
    构建模拟插入工具的缓存

    参数:
        a_length: A 的长度，默认为 2
        save_path: 保存缓存的文件路径
    """
    global _simulation_cache

    # 尝试加载现有缓存
    if os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                _simulation_cache = pickle.load(f)
            print(f"已加载包含 {len(_simulation_cache)} 条记录的缓存")
        except Exception as e:
            print(f"加载缓存时出错: {str(e)}")
            _simulation_cache = {}

    # 计算总组合数（用于进度条）
    if a_length == 1:
        total_combinations = 13 * 13  # 13 * 13
    elif a_length == 2:
        total_combinations = 13 * 13 * 12  # 13 * 13 * 12
    else:
        # 计算 13!/(13-a_length)! * 13
        total_combinations = 13
        for i in range(a_length):
            total_combinations *= (13 - i)

    print(f"将生成 {total_combinations} 种 A 和 x 的组合，每种组合有 {a_length + 1} 个可能的插入位置")
    print(f"预计总缓存条目数: {total_combinations * (a_length + 1)}")

    # 生成并缓存所有可能的组合
    cache_size_before = len(_simulation_cache)

    # 使用进度条显示处理进度
    with tqdm(total=total_combinations, desc="生成缓存") as pbar:
        for A, x in generate_all_Aandx_combinations(a_length):
            # 对所有可能的位置进行缓存
            for pos in range(len(A) + 1):
                key = get_cache_key(A, x, pos)
                if key not in _simulation_cache:
                    result = simulate_insertion_tool(A, x, pos)
                    _simulation_cache[key] = result

            # 更新进度条
            pbar.update(1)

            # 定期保存缓存，避免因意外中断而丢失数据
            if len(_simulation_cache) % 10000 == 0:
                try:
                    with open(save_path, 'wb') as f:
                        pickle.dump(_simulation_cache, f)
                    print(f"已保存中间缓存。当前大小: {len(_simulation_cache)} 条记录")
                except Exception as e:
                    print(f"保存中间缓存时出错: {str(e)}")

    # 保存最终缓存到文件
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(_simulation_cache, f)
        print(f"缓存已保存。新增: {len(_simulation_cache) - cache_size_before}, 总大小: {len(_simulation_cache)} 条记录")
    except Exception as e:
        print(f"保存缓存时出错: {str(e)}")

    return _simulation_cache


def build_all_length_simulation_cache(save_path='../cache/simulation_cache.pkl'):
    """
    构建 A 长度为 1 和 2 的所有可能组合的缓存

    参数:
        save_path: 保存缓存的文件路径
    """
    # 先构建 A 长度为 1 的缓存
    print("开始构建 A 长度为 1 的缓存...")
    build_simulation_cache(a_length=1, save_path=save_path)

    # 再构建 A 长度为 2 的缓存
    print("\n开始构建 A 长度为 2 的缓存...")
    build_simulation_cache(a_length=2, save_path=save_path)

    print("\n所有长度的缓存构建完成！")


def get_simulate_insertion_cache(A, x, pos):
    """
    从缓存中获取模拟插入的结果

    参数:
        A: 卡牌列表
        x: 要插入的卡牌
        pos: 插入位置

    返回:
        如果缓存命中，返回缓存的结果；否则计算并缓存结果
    """
    global _simulation_cache

    key = get_cache_key(A, x, pos)

    # 如果缓存为空，尝试加载
    if not _simulation_cache and os.path.exists('../cache/simulation_cache.pkl'):
        try:
            with open('../cache/simulation_cache.pkl', 'rb') as f:
                _simulation_cache = pickle.load(f)
        except Exception:
            pass

    # 缓存命中
    if key in _simulation_cache:
        return _simulation_cache[key]

    # 缓存未命中，计算结果并缓存
    result = simulate_insertion_tool(A, x, pos)
    _simulation_cache[key] = result

    return result


if __name__ == '__main__':
    # 构建 A 长度为 1 和 2 的所有缓存
    build_all_length_simulation_cache(save_path='../cache/simulation_cache.pkl')