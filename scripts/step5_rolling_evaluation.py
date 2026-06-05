# -*- coding: utf-8 -*-
"""
Step 5：差分还原评估、长程误差分析与滚动预测模拟
主要流程：
模块 C：模拟真实操作工视角，连续展示“历史真实值+刚发生的5分钟预测+未来30分钟预测”的动态滚动过程。
"""
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 配置区
# ==========================================
DATASET_DIR = "data/datasets"
CONFIG_DIR = "configs"
MODEL_DIR = "artifacts/models"
FIGURE_DIR = "artifacts/figures"
MERGED_DATA_DIR = "data/merged"
INPUT_LEN = 30
OUTPUT_LEN = 30
ROLL_STRIDE = 10
num_rolls = 10  # 增加到20次，让拼接线足够长以便观察验证效果
k_start = 161
TARGET_VAR = "碳化取出液温度"
HIDDEN_SIZE = 128
NUM_LAYERS = 3


class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def diff_to_absolute(y_last_abs_norm, y_diff_norm, target_mean, target_std):
    y_last_abs_real = y_last_abs_norm * target_std + target_mean
    y_diff_real = y_diff_norm * target_std
    cum_diff = np.cumsum(y_diff_real, axis=-1)
    y_abs_real = y_last_abs_real + cum_diff
    return y_abs_real


def rolling_evaluation_diff():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 加载标准化参数
    with open(Path(CONFIG_DIR) / "norm_params_diff.json", 'r', encoding='utf-8') as f:
        norm_params = json.load(f)
    target_mean = norm_params['means'][TARGET_VAR]
    target_std = norm_params['stds'][TARGET_VAR]
    # 2. 加载Demo数据
    print("正在加载差分数据集并准备推理...")
    demo_data = np.load(Path(DATASET_DIR) / "demo_data_diff.npz")
    X_demo = torch.FloatTensor(demo_data['X']).to(device)
    Y_demo_diff_norm = demo_data['Y_diff']
    Y_demo_last_abs_norm = demo_data['Y_last_abs']
    # 3. 加载模型
    input_size = X_demo.shape[2]
    model = EnhancedLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_LEN).to(device)
    model.load_state_dict(torch.load(Path(MODEL_DIR) / "best_model_v4_diff.pth", map_location=device))
    model.eval()
    # 4. 读取原始宽表，用于获取连续的绝对温度真实值
    # 保持与数据集构造阶段一致的时间切分边界
    df = pd.read_parquet(Path(MERGED_DATA_DIR) / "merged_wide_table.parquet")
    train_end = 14 * 1440
    val_end = train_end + 2 * 1440
    true_abs_series = df.iloc[val_end:][TARGET_VAR].values
    # ==========================================
    # 模块 A：原有的滚动5分钟评估 (上线真实效果)
    # ==========================================
    print("\n正在模拟滚动预测 (每5分钟更新，差分还原)...")
    rolling_preds_abs, rolling_truths_abs = [], []
    for i in range(0, len(X_demo), 1):
        if i + ROLL_STRIDE > len(Y_demo_diff_norm): break
        x_input = X_demo[i:i + 1]
        with torch.no_grad():
            pred_diff_norm = model(x_input).cpu().numpy()[0]
        true_diff_norm = Y_demo_diff_norm[i]
        last_abs_norm = Y_demo_last_abs_norm[i]
        pred_abs_sequence = diff_to_absolute(last_abs_norm, pred_diff_norm, target_mean, target_std)
        true_abs_sequence = diff_to_absolute(last_abs_norm, true_diff_norm, target_mean, target_std)
        rolling_preds_abs.extend(pred_abs_sequence[:ROLL_STRIDE])
        rolling_truths_abs.extend(true_abs_sequence[:ROLL_STRIDE])
    mae_rolling = np.mean(np.abs(np.array(rolling_preds_abs) - np.array(rolling_truths_abs)))
    print("\n" + "=" * 60)
    print(f"【上线模式】滚动预测 (每{ROLL_STRIDE}分钟更新) 整体指标:")
    print(f"  MAE  (平均绝对误差): {mae_rolling:.4f} ℃")
    print("=" * 60)
    # ==========================================
    # 模块 B：长程预测发散分析 (保留原逻辑)
    # ==========================================
    print("\n" + "=" * 60)
    print("【长程分析】拆解单次预测30分钟的误差发散情况:")
    print("=" * 60)
    fig_long, axes_long = plt.subplots(2, 2, figsize=(16, 10))
    axes_long = axes_long.flatten()
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # for plot_idx, i in enumerate(range(4)):
    # 在模块 C 中随机选中的 k_start 附近取4个窗口
    sample_indices = [k_start, k_start + 2, k_start + 4, k_start + 6]
    for plot_idx, i in enumerate(sample_indices):
        if i >= len(X_demo): break
        x_input = X_demo[i:i + 1]
        with torch.no_grad():
            pred_diff_norm = model(x_input).cpu().numpy()[0]
        true_diff_norm = Y_demo_diff_norm[i]
        last_abs_norm = Y_demo_last_abs_norm[i]
        pred_abs_30min = diff_to_absolute(last_abs_norm, pred_diff_norm, target_mean, target_std)
        true_abs_30min = diff_to_absolute(last_abs_norm, true_diff_norm, target_mean, target_std)
        mae_30 = np.mean(np.abs(pred_abs_30min - true_abs_30min))
        mae_0_5 = np.mean(np.abs(pred_abs_30min[0:5] - true_abs_30min[0:5]))
        mae_5_10 = np.mean(np.abs(pred_abs_30min[5:10] - true_abs_30min[5:10]))
        mae_10_15 = np.mean(np.abs(pred_abs_30min[10:15] - true_abs_30min[10:15]))
        mae_15_30 = np.mean(np.abs(pred_abs_30min[15:30] - true_abs_30min[15:30]))
        print(f"\n第 {i + 1} 次预测 (起始点 Index: {i}):")
        print(f"   ├─ 未来 0-5 分钟 MAE:  {mae_0_5:.4f} ℃  (高置信度区间)")
        print(f"   ├─ 未来 5-10 分钟 MAE: {mae_5_10:.4f} ℃")
        print(f"   ├─ 未来10-15 分钟 MAE: {mae_10_15:.4f} ℃")
        print(f"   └─ 未来15-30 分钟 MAE: {mae_15_30:.4f} ℃  (长程发散区间)")
        print(f"   -> 整体30分钟 MAE:     {mae_30:.4f} ℃")
        ax = axes_long[plot_idx]
        time_steps = np.arange(1, 31)
        ax.plot(time_steps, true_abs_30min, label='真实温度', color='blue', linewidth=2, marker='o', markersize=3)
        ax.plot(time_steps, pred_abs_30min, label='预测温度(30min)', color='red', linestyle='--', linewidth=2,
                marker='x', markersize=3)
        ax.axvspan(1, ROLL_STRIDE, alpha=0.1, color='green', label='高置信区(0-5min)')
        ax.axvspan(ROLL_STRIDE + 10, 30, alpha=0.05, color='red', label='发散区(15-30min)')
        ax.set_title(f'第{i + 1}次独立预测 (30min MAE={mae_30:.2f}℃)', fontweight='bold')
        ax.set_xlabel('预测步长 (分钟)')
        ax.set_ylabel('温度 (°C)')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(FIGURE_DIR) / "longterm_prediction_analysis.png", dpi=150, bbox_inches='tight')
    # ==========================================
    # 模块 C：模拟真实用户 DCS 滚动体验 —— 轨迹拼接版
    # ==========================================
    print("\n" + "=" * 60)
    print("【DCS模拟】实时滚动预测过程演示 (轨迹拼接)")
    print("=" * 60)

    dataset_stride = 5  # Step 4 切分数据集时的 stride，固定为5，不可改
    k_step = ROLL_STRIDE // dataset_stride  # k 每次应跳几个窗口

    # 安全上限也要同步调整
    safe_limit_by_xdemo = len(X_demo) - num_rolls * k_step
    safe_limit_by_series = (len(true_abs_series) - INPUT_LEN - num_rolls * ROLL_STRIDE) // dataset_stride
    safe_limit = max(0, min(safe_limit_by_xdemo, safe_limit_by_series))
    # k_start = random.randint(0, safe_limit)
    # k_start = 153

    base_idx = k_start * 5 + INPUT_LEN
    print(f"\n🎲 随机选中窗口 k_start = {k_start}，T0 原始索引 = {base_idx}")

    fig_real, ax_real = plt.subplots(figsize=(18, 8))

    # 1. 画历史真实值（模型输入）[T0-30, T0]
    history_time = np.arange(base_idx - INPUT_LEN, base_idx)
    history_vals = true_abs_series[base_idx - INPUT_LEN: base_idx]
    ax_real.plot(history_time, history_vals, label='历史真实值 (模型输入)',
                 color='#1f77b4', linewidth=2.5)

    # 2. 画真实发生值（从T0开始，尽可能长，用于验证对比）
    total_sim_len = num_rolls * ROLL_STRIDE + OUTPUT_LEN  # 真实值画长一点
    sim_true_time = np.arange(base_idx, base_idx + total_sim_len)
    sim_true_vals = true_abs_series[base_idx: base_idx + total_sim_len]
    ax_real.plot(sim_true_time, sim_true_vals, label='真实发生值',
                 color='#1f77b4', linewidth=2.5, linestyle='-', alpha=0.9)

    # 3. 轨迹拼接：维护一条与真实值时间对齐的连续预测线
    stitched_pred_time = []  # 时间轴
    stitched_pred_vals = []  # 预测值
    prev_preds = []  # 保留完整预测用于验证打印

    for r in range(num_rolls):
        k = k_start + r * k_step  # ← 关键修正：k 按 ROLL_STRIDE 步进
        current_idx = k * dataset_stride + INPUT_LEN

        x_input = X_demo[k:k + 1]
        with torch.no_grad():
            pred_diff_norm = model(x_input).cpu().numpy()[0]
        last_abs_norm = Y_demo_last_abs_norm[k]
        pred_abs_30 = diff_to_absolute(last_abs_norm, pred_diff_norm, target_mean, target_std)
        prev_preds.append(pred_abs_30)

        # 拼接：取前 ROLL_STRIDE 分钟，与真实值时间对齐
        seg_time = np.arange(current_idx, current_idx + ROLL_STRIDE)
        seg_vals = pred_abs_30[:ROLL_STRIDE]
        stitched_pred_time.extend(seg_time)
        stitched_pred_vals.extend(seg_vals)

        # 打印验证（每2次打印一次）
        if r > 0 and r % 2 == 0:
            prev_pred_seg = prev_preds[r - 1][:ROLL_STRIDE]
            true_seg = true_abs_series[current_idx - ROLL_STRIDE: current_idx]
            mae_seg = np.mean(np.abs(prev_pred_seg - true_seg))
            print(f"T+{r * ROLL_STRIDE} 分钟 | 过去{ROLL_STRIDE}分钟验证 MAE: {mae_seg:.4f} ℃")

        # 4. 画拼接轨迹
    ax_real.plot(stitched_pred_time, stitched_pred_vals,
                 label=f'拼接预测轨迹 (每{ROLL_STRIDE}分钟更新，与真实值对齐)',
                 color='#d62728', linewidth=2.5, linestyle='-', alpha=0.85)

    # 5. 画最新30分钟远景
    final_pred_time = np.arange(current_idx, current_idx + OUTPUT_LEN)
    final_pred_vals = prev_preds[-1]
    ax_real.plot(final_pred_time, final_pred_vals,
                 label='最新30分钟远景预测',
                 color='#d62728', linewidth=2, linestyle='--', alpha=0.5)

    # 图表美化
    ax_real.set_title('模拟真实用户滚动预测体验 (轨迹拼接版)', fontsize=16, fontweight='bold')
    ax_real.set_xlabel('时间序列 (分钟)', fontsize=12)
    ax_real.set_ylabel('温度 (°C)', fontsize=12)
    ax_real.axvline(x=base_idx, color='gray', linestyle='--', alpha=0.7, label='预测起始点 (T0)')
    ax_real.legend(fontsize=11, loc='best')
    ax_real.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(FIGURE_DIR) / "realtime_rolling_simulation_stitched.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nDCS模拟对比图已保存至: {Path(FIGURE_DIR) / 'realtime_rolling_simulation_stitched.png'}")


if __name__ == "__main__":
    rolling_evaluation_diff()
