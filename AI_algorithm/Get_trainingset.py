import json

from AI_algorithm.brute_force import recursive_StrategyAndScore



from AI_algorithm.tool.tool import deal_cards_tool


def has_duplicates(lst):
    return len(lst) != len(set(lst))


h=0
def generate_training_data(num_samples=50000):
    dataset = []
    global h
    for i in range(num_samples):



        A, B = deal_cards_tool()  # 初始A, B   A, B 都是 list<int>


        max_score, best_moves = recursive_StrategyAndScore(A, B)
        print(f"{i} of {num_samples}")
        # bestmoves的长度未必都是3  因为当B中就算再出牌也不能得分时 递归不会考虑加入移动到策略中
        # 这里也排除了A,B完全不能得分的情况
        # if len(best_moves) != 3:
        #     h+=1
        #     print(f"跳过了len(best_moves)!=3的情况{h}次")
        #     print(f"A:{A} B:{B}")
        #
        #     print(f"分数{max_score}, 最佳移动:{best_moves}")
        #     continue

        whole=[0,1,2]
        if len(best_moves)==0:
            best_moves=[[0,0],[1,0],[2,0]]

        if len(best_moves)==1:
            i=best_moves[0][0]
            whole.remove(i)
            best_moves.append([whole[0], 0],[whole[1],0])

            best_moves.append([whole[0],0])
        if len(best_moves)==2:
            i=best_moves[0][0]
            j=best_moves[1][0]
            whole.remove(i)
            whole.remove(j)
            best_moves.append([whole[0],0])




        dataset.append({"A": A, "B": B, "max_score": max_score, "best_moves": best_moves})

    with open("json/data_raw.json", "w") as f:
        json.dump(dataset, f, indent=4)


generate_training_data()
