import unittest


from AI_algorithm.tool.tool import simulate_insertion_tool, load_best_genome


class TestSimulateInsertion(unittest.TestCase):

    def test_simulate_insertion1(self):
        """
        测试 simulate_insertion_tool 函数并验证每个位置的相关信息
        """
        A = [1, 2, 3, 4, 5, 6]
        x = 1

        # 定义期望的测试结果（根据实际函数逻辑填写）
        expected_results = {
            0: {"score": 2, "removed_len": 2, "new_len": 5, "match_found": 1, "updated_A": [2,3,4,5,6]},
            1: {"score": 2, "removed_len": 2, "new_len": 5, "match_found": 1, "updated_A": [2,3,4,5,6]},
            2: {"score": 4, "removed_len": 3, "new_len": 4, "match_found": 1, "updated_A": [3,4,5,6]},
            3: {"score": 7, "removed_len": 4, "new_len": 3, "match_found": 1, "updated_A": [4,5,6]},
            4: {"score": 11, "removed_len": 5, "new_len": 2, "match_found": 1, "updated_A":[5,6]},
            5: {"score": 16, "removed_len": 6, "new_len": 1, "match_found": 1, "updated_A":[6]},
            6: {"score": 22, "removed_len": 7, "new_len": 0, "match_found": 1, "updated_A":[]},
        }

        for pos in range(0, len(A) + 1):
            # 调用被测函数
            score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(A, x, pos)

            # 获取当前插入位置的期望结果
            expected = expected_results[pos]

            # 验证返回值是否符合预期
            self.assertEqual(score, expected["score"], f"位置 {pos} 的分数不匹配")
            self.assertEqual(removed_len, expected["removed_len"], f"位置 {pos} 的移除长度不匹配")
            self.assertEqual(new_len, expected["new_len"], f"位置 {pos} 的新长度不匹配")
            self.assertEqual(match_found, expected["match_found"], f"位置 {pos} 的匹配状态不匹配")
            self.assertEqual(updated_A, expected["updated_A"], f"位置 {pos} 的更新数组不匹配")

    def test_simulate_insertion2(self):
        """
        测试 simulate_insertion_tool 函数并验证每个位置的相关信息
        """
        A = [7, 2, 3, 4, 5, 6]
        x = 1

        # 定义期望的测试结果（根据实际函数逻辑填写）
        expected_results = {
            0: {"score": 0, "removed_len": 0, "new_len": 7, "match_found": 0, "updated_A": [1,7,2,3,4,5,6]},
            1: {"score": 0, "removed_len": 0, "new_len": 7, "match_found": 0, "updated_A": [7,1,2,3,4,5,6]},
            2: {"score": 0, "removed_len": 0, "new_len": 7, "match_found": 0, "updated_A": [7,2,1,3,4,5,6]},
            3: {"score": 0, "removed_len": 0, "new_len": 7, "match_found": 0, "updated_A": [7,2,3,1,4,5,6]},
            4: {"score": 0, "removed_len": 0, "new_len": 7, "match_found": 0, "updated_A": [7,2,3,4,1,5,6]},
            5: {"score": 0, "removed_len": 0, "new_len": 7, "match_found": 0, "updated_A": [7,2,3,4,5,1,6]},
            6: {"score": 0, "removed_len": 0, "new_len": 7, "match_found": 0, "updated_A": [7,2,3,4,5,6,1]},
        }

        for pos in range(0, len(A) + 1):
            # 调用被测函数
            score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(A, x, pos)

            # 获取当前插入位置的期望结果
            expected = expected_results[pos]

            # 验证返回值是否符合预期
            self.assertEqual(score, expected["score"], f"位置 {pos} 的分数不匹配")
            self.assertEqual(removed_len, expected["removed_len"], f"位置 {pos} 的移除长度不匹配")
            self.assertEqual(new_len, expected["new_len"], f"位置 {pos} 的新长度不匹配")
            self.assertEqual(match_found, expected["match_found"], f"位置 {pos} 的匹配状态不匹配")
            self.assertEqual(updated_A, expected["updated_A"], f"位置 {pos} 的更新数组不匹配")


if __name__ == '__main__':
    unittest.main()
