import unittest


from AI_algorithm.tool.tool import simulate_insertion_tool, load_best_genome


class TestSimulateInsertion(unittest.TestCase):

    def test_simulate_insertion(self):
        """
        测试 simulate_insertion_tool 函数并输出每个位置的相关信息
        """
        A = [1, 2, 3, 6, 2, 9]
        x = 1
        for pos in range(1, len(A) + 1):
            score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(A, x, pos)
            if match_found:
                print(f"插入位置: {pos}")
                print(f"被移除的数组长度: {removed_len}")
                print(f"更新后的数组: {updated_A}")
                print("-" * 30)

    def test_simulate_insertion2(self):
        """
        针对另一组数据测试 simulate_insertion_tool 并输出相关信息
        """
        B = [4, 11, 5, 3, 7, 1]
        x = 7
        for pos in range(1, len(B) + 1):
            score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(B, x, pos)
            if match_found:
                print(f"插入位置: {pos}")
                print(f"被移除的数组长度: {removed_len}")
                print(f"更新后的数组: {updated_A}")
                print("-" * 30)

    def test_simulate_insertion3(self):
        """
        对第三组数据进行插入并输出结果
        """
        B = [2, 0, 1]
        x = 8
        for pos in range(1, len(B) + 1):
            score, removed_len, new_len, match_found, updated_A = simulate_insertion_tool(B, x, pos)
            if match_found:
                print(f"插入位置: {pos}")
                print(f"被移除的数组长度: {removed_len}")
                print(f"更新后的数组: {updated_A}")
                print("-" * 30)




if __name__ == '__main__':
    unittest.main()
