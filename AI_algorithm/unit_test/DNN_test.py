import unittest

from AI_algorithm.Deep_Neural_Network import DNNpredict
from AI_algorithm.brute_force import recursive_StrategyAndScore
from AI_algorithm.tool.tool import deal_cards_tool, calculate_score_by_strategy


class TestPrediction_Of_DNN(unittest.TestCase):
    def setUp(self):
        """
        在每个测试用例运行前执行的初始化操作。
        可以在这里加载模型、初始化数据等。
        """
        self.model_path = "../trained/move_predictor.pth"

    def test_prediction_vs_recursive_strategy(self):
        """
        测试神经网络预测与递归算法的结果对比。
        """
        for _ in range(5):  # 进行5组测试
            A, B = deal_cards_tool()
            best_moves, score0 = DNNpredict(A, B, model_path=self.model_path)
            score = calculate_score_by_strategy(A, B, best_moves)

            # 测试1: 预测得分与计算得分是否匹配
            with self.subTest(A=A, B=B, best_moves=best_moves, score=score, score0=score0):
                self.assertEqual(score, score0,
                                 f"预测得分 ({score0}) 与计算得分 ({score}) 不匹配！")

            # 测试2: 神经网络预测得分不应大于递归算法得分
            score1, move1 = recursive_StrategyAndScore(A, B)
            with self.subTest(A=A, B=B, score1=score1, score0=score0):
                self.assertGreaterEqual(score1, score0,
                                        f"预测出了比真实值还大的值！预测得分: {score0}, 递归得分: {score1}")

            # 测试3: best_moves 格式校验
            first_elements = [move[0] for move in best_moves]
            with self.subTest(best_moves=best_moves, first_elements=first_elements):
                self.assertEqual(set(first_elements), {0, 1, 2},
                                 f"best_moves 格式错误: {best_moves}，子数组的第一个元素必须是 0, 1, 2 且互异")

            # 打印测试结果（可选）
            print(f"\nA: {A}")
            print(f"B: {B}")
            print(f"预测最佳移动: {best_moves}")
            print(f"预测最佳得分: {score0}")
            print(f"预测策略实际得分: {score}")
            print(f"递归算法得分: {score1}")
            print(f"递归算法移动: {move1}")
            print(f"得分差: {score1 - score} (正数表示递归更优)")

if __name__ == "__main__":
    unittest.main()