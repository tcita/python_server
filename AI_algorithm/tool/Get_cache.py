import json
import pickle


def preprocess_cache_file(json_file_path, output_file_path=None):
    """
    将原始JSON文件预处理为优化的字典格式并保存
    """
    # 读取原始JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 转换为优化的查询格式 - 使用元组作为键
    optimized_cache = {}
    for entry in data:
        # 将列表转换为元组，因为元组是可哈希的，可以直接作为字典键
        a_tuple = tuple(entry["A"])
        b_tuple = tuple(entry["B"])
        key = (a_tuple, b_tuple)

        optimized_cache[key] = {
            "max_score": entry["max_score"],
            "best_moves": entry["best_moves"]
        }

    # 保存优化后的数据结构
    if output_file_path:
        with open(output_file_path, 'wb') as f:
            pickle.dump(optimized_cache, f)

    return optimized_cache


class FastCacheQuery:
    def __init__(self, cache_data):
        """
        初始化查询系统
        cache_data可以是预处理后的字典或JSON文件路径
        """
        if isinstance(cache_data, str):
            if cache_data.endswith('.pkl'):
                # 加载预处理的pickle文件
                with open(cache_data, 'rb') as f:
                    self.cache = pickle.load(f)
            else:
                # 假设是原始JSON文件
                self.cache = preprocess_cache_file(cache_data)
        else:
            # 已经是处理好的字典
            self.cache = cache_data

    def query(self, A, B):
        """查询缓存"""
        key = (tuple(A), tuple(B))
        result = self.cache.get(key)

        if result:
            return {
                'hit': True,
                'max_score': result['max_score'],
                'best_moves': result['best_moves']
            }
        return {'hit': False}


# 使用示例
if __name__ == "__main__":
    # 预处理文件（只需运行一次）
    optimized_cache = preprocess_cache_file("../json/extreme_scores_uniq.json", "../extreme_cache/optimized_cache.pkl")
    #
    # # 使用预处理后的数据进行快速查询
    # cache_query = FastCacheQuery("optimized_cache.pkl")
    # # 或者直接使用：cache_query = FastCacheQuery(optimized_cache)
    #
    # # 查询示例
    # A = [9, 13, 7, 12, 11, 10]
    # B = [12, 11, 10]
    # result = cache_query.query(A, B)
    #
    # if result['hit']:
    #     print(f"缓存命中! 最大分数: {result['max_score']}")
    #     print(f"最佳移动: {result['best_moves']}")
    # else:
    #     print("缓存未命中")