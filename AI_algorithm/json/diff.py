import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


def load_json_data(file_path):
    """
    加载JSON数据
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_dataset_features(data):
    """
    提取数据集特征，使用百分比计算
    """
    total_samples = len(data)
    A_values = [val for sample in data for val in sample['A']]
    B_values = [val for sample in data for val in sample['B']]
    max_scores = [sample['max_score'] for sample in data]
    best_moves = [move for sample in data for move in sample['best_moves']]

    def calculate_distribution_percentages(values):
        unique, counts = np.unique(values, return_counts=True)
        percentages = counts / len(values) * 100
        return dict(zip(map(str, unique), percentages))

    return {
        'total_samples': total_samples,
        'A_stats': {
            'unique_count': len(set(A_values)),
            'mean': np.mean(A_values),
            'median': np.median(A_values),
            'std': np.std(A_values),
            'distribution_percentages': calculate_distribution_percentages(A_values)
        },
        'B_stats': {
            'unique_count': len(set(B_values)),
            'mean': np.mean(B_values),
            'median': np.median(B_values),
            'std': np.std(B_values),
            'distribution_percentages': calculate_distribution_percentages(B_values)
        },
        'max_score_stats': {
            'mean': np.mean(max_scores),
            'median': np.median(max_scores),
            'std': np.std(max_scores),
            'distribution_percentages': calculate_distribution_percentages(max_scores)
        },
        'best_moves_stats': {
            'total_moves': len(best_moves),
            'unique_moves': len(set(tuple(move) for move in best_moves)),
            'move_frequencies_percentages': {
                str(move): count / len(best_moves) * 100
                for move, count in
                dict(zip(*np.unique([tuple(move) for move in best_moves], return_counts=True))).items()
            }
        }
    }


def compare_datasets(dataset1, dataset2):
    """
    比较两个数据集的特征，使用百分比差异
    """
    differences = {
        'total_samples': {
            'dataset1': dataset1['total_samples'],
            'dataset2': dataset2['total_samples']
        }
    }

    # 比较A值分布百分比
    a_diff = {}
    all_a_values = set(list(dataset1['A_stats']['distribution_percentages'].keys()) +
                       list(dataset2['A_stats']['distribution_percentages'].keys()))
    for val in all_a_values:
        percent1 = dataset1['A_stats']['distribution_percentages'].get(val, 0)
        percent2 = dataset2['A_stats']['distribution_percentages'].get(val, 0)
        if abs(percent1 - percent2) > 0.5:  # 超过0.5%的差异
            a_diff[val] = {
                'dataset1_percent': round(percent1, 2),
                'dataset2_percent': round(percent2, 2),
                'difference': round(percent1 - percent2, 2)
            }
    if a_diff:
        differences['A_distribution_differences'] = a_diff

    # 类似地比较B值和最佳移动
    b_diff = {}
    all_b_values = set(list(dataset1['B_stats']['distribution_percentages'].keys()) +
                       list(dataset2['B_stats']['distribution_percentages'].keys()))
    for val in all_b_values:
        percent1 = dataset1['B_stats']['distribution_percentages'].get(val, 0)
        percent2 = dataset2['B_stats']['distribution_percentages'].get(val, 0)
        if abs(percent1 - percent2) > 0.5:
            b_diff[val] = {
                'dataset1_percent': round(percent1, 2),
                'dataset2_percent': round(percent2, 2),
                'difference': round(percent1 - percent2, 2)
            }
    if b_diff:
        differences['B_distribution_differences'] = b_diff

    # 比较最佳移动频率
    moves_diff = {}
    all_moves = set(list(dataset1['best_moves_stats']['move_frequencies_percentages'].keys()) +
                    list(dataset2['best_moves_stats']['move_frequencies_percentages'].keys()))
    for move in all_moves:
        percent1 = dataset1['best_moves_stats']['move_frequencies_percentages'].get(move, 0)
        percent2 = dataset2['best_moves_stats']['move_frequencies_percentages'].get(move, 0)
        if abs(percent1 - percent2) > 0.5:
            moves_diff[move] = {
                'dataset1_percent': round(percent1, 2),
                'dataset2_percent': round(percent2, 2),
                'difference': round(percent1 - percent2, 2)
            }
    if moves_diff:
        differences['moves_frequency_differences'] = moves_diff

    return differences


def main():
    # 加载数据
    transformer_error_cases = load_json_data('transformer_error_cases.json')
    data_raw = load_json_data('data_raw.json')

    # 提取特征
    error_cases_features = extract_dataset_features(transformer_error_cases)
    raw_data_features = extract_dataset_features(data_raw)

    # 比较数据集
    differences = compare_datasets(error_cases_features, raw_data_features)

    # 打印差异
    print("数据集差异:")
    print(json.dumps(differences, indent=2))


if __name__ == '__main__':
    main()