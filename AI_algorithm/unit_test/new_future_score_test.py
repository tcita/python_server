import unittest
# 确保这个导入路径是正确的，并且Python可以找到这个模块
from AI_algorithm.tool.tool import calculate_future_score_new

# 1. 创建一个类，并让它继承 unittest.TestCase
class TestCalculateFutureScore(unittest.TestCase):

    # 2. 将所有测试函数缩进，使其成为这个类的方法
    def test_b_has_duplicates(self):
        """
        测试情况1：当列表B有重复元素时。此逻辑不受影响。
        """
        A = [10, 20, 30]
        B = [40, 50, 40]
        expected = 190
        self.assertEqual(calculate_future_score_new(A, B), expected, "当B有重复元素时，计算错误")

    def test_no_intersection(self):
        """
        测试情况2：当A和B没有交集时。此逻辑不受影响。
        """
        A = [1, 2, 3]
        B = [4, 5, 6]
        self.assertEqual(calculate_future_score_new(A, B), 0, "无交集时应返回0")

    def test_user_provided_example(self):
        """
        测试您在问题中提供的核心示例 A=[2,1,3], B=[1,3]。
        """
        A = [2, 1, 3]
        B = [1, 3]
        # 过程:
        # 1. 处理 item=1 (idx=1):
        #    left_sum = sum([2,1])+1 = 4
        #    right_sum = sum([1,3])+1 = 5
        #    score += max(4,5) -> score = 5.
        #    删除右侧, A 变为 [2].
        # 2. 处理 item=3: 3不在A中，跳过.
        # 最终结果: 5
        self.assertEqual(calculate_future_score_new(A, B), 5, "用户提供的核心示例计算错误")
        self.assertEqual(A, [2, 1, 3], "函数不应修改原始输入列表A")

    def test_complex_scenario_with_multiple_deletions(self):
        """
        测试一个更复杂的场景，其中多次删除会影响后续元素的处理。
        """
        A = [100, 20, 5, 1, 30]
        B = [20, 30]
        # 过程:
        # 1. 处理 item=20 (idx=1):
        #    left_sum = sum([100,20])+20 = 140
        #    right_sum = sum([20,5,1,30])+20 = 76
        #    score += max(140,76) -> score = 140.
        #    删除左侧, A 变为 [5,1,30].
        # 2. 处理 item=30 (在新A中idx=2):
        #    left_sum = sum([5,1,30])+30 = 66
        #    right_sum = sum([30])+30 = 60
        #    score += max(66,60) -> score = 140 + 66 = 206.
        #    删除左侧, A 变为 [].
        # 最终结果: 206
        self.assertEqual(calculate_future_score_new(A, B), 206, "复杂场景计算错误")

    def test_right_side_deletion_first(self):
        """
        测试优先删除右侧部分的场景。
        """
        A = [10, 20, 30, 40, 50]
        B = [20, 50]
        # 过程:
        # 1. 处理 item=20 (idx=1):
        #    left_sum = sum([10,20])+20 = 50
        #    right_sum = sum([20,30,40,50])+20 = 160
        #    score += max(50,160) -> score = 160.
        #    删除右侧, A 变为 [10].
        # 2. 处理 item=50: 50不在A中，跳过.
        # 最终结果: 160
        self.assertEqual(calculate_future_score_new(A, B), 160, "右侧删除场景计算错误")

    def test_empty_lists(self):
        """
        测试边界条件：输入列表为空。此逻辑不受影响。
        """
        self.assertEqual(calculate_future_score_new([], [1, 2]), 0, "当A为空列表时应返回0")
        self.assertEqual(calculate_future_score_new([1, 2], []), 0, "当B为空列表时应返回0")
        self.assertEqual(calculate_future_score_new([], []), 0, "当A和B都为空列表时应返回0")

    def test_identical_lists(self):
        """
        测试边界条件：A和B完全相同。
        """
        A = [10, 20, 30]
        B = [10, 20, 30]
        # 过程:
        # 1. 处理 item=10 (idx=0):
        #    left_sum = sum([10])+10 = 20
        #    right_sum = sum([10,20,30])+10 = 70
        #    score += max(20,70) -> score = 70.
        #    删除右侧, A 变为 [].
        # 2. 处理20,30: 不在A中，跳过.
        # 最终结果: 70
        self.assertEqual(calculate_future_score_new(A, B), 70, "相同列表场景计算错误")

# 这部分代码保持不变
if __name__ == "__main__":
    unittest.main(verbosity=2)