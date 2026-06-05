# -*- coding: utf-8 -*-
"""
Step 3 升级版：领先指标挖掘机 (基于原 step3_feature_selection_lag_correlation)
核心目标：
在包含13个特征的数据集中，寻找领先于目标变量(碳化取出液温度)的"超前预告牌"。
过滤掉滞后指标(Lag < 0，无法用于预测)，保留同步和领先指标，重点挑出具备超前预测能力的强相关特征。
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================
# 配置区
# ==========================================
MERGED_DATA_DIR = "data/merged"
MERGED_FILENAME = "merged_wide_table.parquet"  # 刚才合并好的13特征宽表
CONFIG_DIR = "configs"
FIGURE_DIR = "artifacts/figures"
TARGET_VAR = "碳化取出液温度"  # 我们的目标预测变量
MAX_LAG = 30  # 最大考察滞后阶数（往前看30分钟）
# 特征筛选阈值
CORR_THRESHOLD = 0.35  # 相关系数绝对值阈值，低于此值的关联太弱，剔除


# ==========================================
# 核心分析函数
# ==========================================
def discover_lead_indicators(df, target_var, max_lag, corr_threshold):
    print(f"正在寻找目标变量 [{target_var}] 的领先指标 (最大前瞻 {max_lag} 分钟)...\n")
    features = [col for col in df.columns if col != target_var]
    results = []
    for feat in features:
        # 计算从 -max_lag 到 +max_lag 的交叉相关性
        # Lag > 0: 特征领先目标 (重点寻找！); Lag < 0: 特征滞后目标 (剔除!); Lag = 0: 同步
        corrs = [df[target_var].corr(df[feat].shift(lag)) for lag in range(-max_lag, max_lag + 1)]
        # 找到绝对值最大的相关性及其对应的滞后阶数
        best_idx = np.argmax(np.abs(corrs))
        best_lag = list(range(-max_lag, max_lag + 1))[best_idx]
        best_corr = corrs[best_idx]
        # 记录结果
        results.append({
            'feature': feat,
            'best_corr': best_corr,
            'abs_corr': abs(best_corr),
            'best_lag': best_lag,
            'type': '领先' if best_lag > 0 else ('同步' if best_lag == 0 else '滞后')
        })
    # 转为 DataFrame 方便分析
    result_df = pd.DataFrame(results)
    # 核心筛选逻辑：挑选有预测基础的特征
    # 1. 剔除滞后指标 (Lag < 0)：它比目标反应还慢，无法用于预测未来
    # 2. 剔除弱相关 (abs_corr < threshold)：关联太弱，是噪音
    valid_df = result_df[(result_df['best_lag'] >= 0) & (result_df['abs_corr'] >= corr_threshold)].copy()
    # 排序规则：按绝对相关性降序排，确保核心强相关变量入选
    valid_df = valid_df.sort_values(by='abs_corr', ascending=False).reset_index(drop=True)
    return result_df, valid_df


def visualize_indicator_types(result_df, target_var):
    """可视化特征的分类：领先、同步、滞后"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'领先': 'green', '同步': 'blue', '滞后': 'red'}
    for typ, group in result_df.groupby('type'):
        ax.scatter(group['best_lag'], group['abs_corr'],
                   c=colors[typ], label=typ, s=100, alpha=0.8)
        for _, row in group.iterrows():
            ax.annotate(row['feature'], (row['best_lag'], row['abs_corr']),
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax.set_xlabel("最佳滞后阶数 (分钟) \n (<0:特征滞后于目标 | >0:特征领先于目标)", fontsize=11)
    ax.set_ylabel("最大皮尔逊相关系数 (绝对值)", fontsize=11)
    ax.set_title(f"谁是 [{target_var}] 的领先指标？", fontweight='bold', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot_path = Path(FIGURE_DIR) / "lead_indicator_scatter.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"领先指标散点图已保存至: {plot_path}")


def save_feature_config(valid_df, target_var):
    """生成供 Step 4 使用的特征配置文件，并打印最终入选名单"""
    print("\n" + "=" * 60)
    print(f"预测特征筛选结果 (目标: {target_var})")
    print("（已剔除滞后指标和弱相关噪音）")
    print("=" * 60)
    feature_config = {}
    for idx, row in valid_df.iterrows():
        feat = row['feature']
        lag = int(row['best_lag'])
        corr = row['best_corr']
        typ = row['type']
        # 添加到配置字典
        feature_config[feat] = lag
        # 打印入选理由
        tag = "领先指标" if typ == '领先' else "同步指标"
        print(f"  {tag} [{feat}]")
        print(f"     -> 相关性: {corr:.4f}, 领先时间: {lag} 分钟")
    # 保存配置，供 Step 4 读取
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_path = Path(CONFIG_DIR) / "feature_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=4)
    print(f"\n特征配置已保存至: {config_path}")
    print(f"共入选 {len(feature_config)} 个特征，其中领先指标 {len(valid_df[valid_df['type'] == '领先'])} 个。")


# ==========================================
# 主执行流程
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("开始执行 Step 3: 领先指标挖掘 (特征滞后相关性分析)")
    print("=" * 60)
    file_path = Path(MERGED_DATA_DIR) / MERGED_FILENAME
    if not file_path.exists():
        print(f"找不到宽表文件: {file_path}，请先执行数据合并步骤。")
    else:
        df = pd.read_parquet(file_path)
        print(f"成功读取宽表，共 {len(df.columns) - 1} 个候选特征变量。\n")
        # 1. 挖掘分析
        all_result_df, valid_result_df = discover_lead_indicators(df, TARGET_VAR, MAX_LAG, CORR_THRESHOLD)
        # 2. 可视化分类
        visualize_indicator_types(all_result_df, TARGET_VAR)
        # 3. 保存配置与打印结果
        save_feature_config(valid_result_df, TARGET_VAR)
