import json
import unittest

jsonfilename = "../json/data_raw.json"

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        """加载测试数据"""
        with open(jsonfilename, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def test_best_moves_validity(self):
        """测试所有样本的best_moves是否满足条件"""
        invalid_samples = []  # 存储不符合条件的样本

        for idx, sample in enumerate(self.data):
            best_moves = sample.get("best_moves", [])

            # 检查是否为空或长度不为3
            if not best_moves or len(best_moves) != 3:
                invalid_samples.append({
                    "index": idx,
                    "A": sample.get("A"),
                    "B": sample.get("B"),
                    "best_moves": best_moves,
                    "error_type": "empty" if not best_moves else "length_mismatch"
                })
                continue

            # 提取所有内层数组的第一个元素
            first_elements = {move[0] for move in best_moves if isinstance(move, list) and move}

            # 检查是否包含 0, 1, 2（顺序不定）
            if first_elements != {0, 1, 2}:
                invalid_samples.append({
                    "index": idx,
                    "A": sample.get("A"),
                    "B": sample.get("B"),
                    "best_moves": best_moves,
                    "error_type": "first_elements_invalid"
                })

        # 如果存在不符合条件的样本，测试失败并输出详细信息
        if invalid_samples:
            error_msg = "发现以下不符合条件的样本:\n"
            for item in invalid_samples:
                error_msg += f"样本索引: {item['index']}\n"
                error_msg += f"A: {item['A']}\n"
                error_msg += f"B: {item['B']}\n"
                error_msg += f"Best Moves: {item['best_moves']}\n"
                error_msg += f"问题类型: {item['error_type']}\n"
                error_msg += "-" * 40 + "\n"
            self.fail(error_msg)
        else:
            print(jsonfilename + " 所有样本的best_moves均符合要求。")


if __name__ == "__main__":
    unittest.main()
