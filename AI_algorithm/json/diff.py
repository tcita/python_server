import json
import numpy as np
import scipy.stats as stats


def advanced_dataset_analysis(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 提取详细特征
    A_values = [val for sample in data for val in sample['A']]
    B_values = [val for sample in data for val in sample['B']]
    max_scores = [sample['max_score'] for sample in data]

    analysis = {
        'total_samples': len(data),

        # 数值分布特征（保持原有计算）
        'A_distribution': {
            'mean': float(np.mean(A_values)),
            'median': float(np.median(A_values)),
            'std': float(np.std(A_values)),
            'skewness': float(stats.skew(A_values)),
            'kurtosis': float(stats.kurtosis(A_values))
        },
        'B_distribution': {
            'mean': float(np.mean(B_values)),
            'median': float(np.median(B_values)),
            'std': float(np.std(B_values)),
            'skewness': float(stats.skew(B_values)),
            'kurtosis': float(stats.kurtosis(B_values))
        },
        'max_score_distribution': {
            'mean': float(np.mean(max_scores)),
            'median': float(np.median(max_scores)),
            'std': float(np.std(max_scores)),
            'skewness': float(stats.skew(max_scores)),
            'kurtosis': float(stats.kurtosis(max_scores))
        },

        # 使用百分比计算值频率
        'A_value_percentages': {},
        'B_value_percentages': {},

        # 移动频率百分比
        'best_moves_percentages': {}
    }

    # 计算A值百分比
    A_value_counts = {}
    for val in A_values:
        A_value_counts[val] = A_value_counts.get(val, 0) + 1
    for val, count in A_value_counts.items():
        analysis['A_value_percentages'][str(val)] = count / len(A_values) * 100

    # 计算B值百分比
    B_value_counts = {}
    for val in B_values:
        B_value_counts[val] = B_value_counts.get(val, 0) + 1
    for val, count in B_value_counts.items():
        analysis['B_value_percentages'][str(val)] = count / len(B_values) * 100

    # 计算最佳移动百分比
    moves_count = {}
    total_moves = 0
    for sample in data:
        for move in sample['best_moves']:
            move_key = str(tuple(move))
            moves_count[move_key] = moves_count.get(move_key, 0) + 1
            total_moves += 1

    for move, count in moves_count.items():
        analysis['best_moves_percentages'][move] = count / total_moves * 100

    return analysis


# 分析两个数据集
error_cases_advanced = advanced_dataset_analysis('transformer_error_cases.json')
raw_data_advanced = advanced_dataset_analysis('data_raw.json')


# 比较分析结果
def compare_datasets(error_cases, raw_data):
    differences = {
        'sample_size': {
            'error_cases': error_cases['total_samples'],
            'raw_data': raw_data['total_samples']
        }
    }

    # 比较值百分比
    for category in ['A_value_percentages', 'B_value_percentages', 'best_moves_percentages']:
        diff = {}
        all_keys = set(list(error_cases[category].keys()) + list(raw_data[category].keys()))
        for key in all_keys:
            error_percent = error_cases[category].get(key, 0)
            raw_percent = raw_data[category].get(key, 0)
            if abs(error_percent - raw_percent) > 0.1:  # 超过0.1%的差异
                diff[key] = {
                    'error_cases': round(error_percent, 2),
                    'raw_data': round(raw_percent, 2),
                    'difference': round(error_percent - raw_percent, 2)
                }
        if diff:
            differences[category] = diff

    return differences


# 输出差异
differences = compare_datasets(error_cases_advanced, raw_data_advanced)
print("Dataset Differences (Percentage-based):")
print(json.dumps(differences, indent=2))