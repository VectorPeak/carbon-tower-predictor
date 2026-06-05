# -*- coding: utf-8 -*-
"""
Step 4 终极版：差分预测数据集构造
核心思想：不预测绝对温度，预测温度的变化量(差分)。
这能极大减轻模型学习历史基线的负担，让模型把全部精力放在学习"扰动响应"上。
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ==========================================
# 配置区
# ==========================================
MERGED_DATA_DIR = "data/merged"
MERGED_FILENAME = "merged_wide_table.parquet"
CONFIG_DIR = "configs"
DATASET_DIR = "data/datasets"
INPUT_LEN = 30
OUTPUT_LEN = 30
STRIDE = 5
TARGET_VAR = "碳化取出液温度"


# ==========================================
# 动量与滞后处理 (复用之前逻辑)
# ==========================================
def add_momentum_features(df, target_var):
    features = [col for col in df.columns if col != target_var]
    for feat in features:
        df[f"{feat}_diff1"] = df[feat].diff(periods=1)
        df[f"{feat}_diff3"] = df[feat].diff(periods=3)
    return df


def apply_lag_shifts(df, config_path, target_var):
    with open(config_path, 'r', encoding='utf-8') as f:
        feature_config = json.load(f)
    for feat, lag in feature_config.items():
        if feat == target_var: continue
        if lag > 0:
            df[feat] = df[feat].shift(lag)
    return df


def split_timeseries(df):
    train_end = 14 * 1440
    val_end = train_end + 2 * 1440
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def normalize_and_save_params(df_train, df_val, df_demo, features):
    means = df_train[features].mean()
    stds = df_train[features].std()
    stds[stds == 0] = 1e-6
    norm_params = {"means": means.to_dict(), "stds": stds.to_dict(), "features_order": features}
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(Path(CONFIG_DIR) / "norm_params_diff.json", 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, ensure_ascii=False, indent=4)

    def scale(d): return (d[features] - means) / stds

    return scale(df_train), scale(df_val), scale(df_demo)


# ==========================================
# 核心：差分滑动窗口
# ==========================================
def create_diff_sliding_windows(df, input_len, output_len, stride, target_var):
    data_X = df.drop(columns=[target_var]).values
    data_Y_absolute = df[[target_var]].values  # 拿到绝对温度
    X_list, Y_diff_list, Y_last_abs_list = [], [], []
    for i in range(0, len(df) - input_len - output_len + 1, stride):
        x_win = data_X[i: i + input_len]
        # 🚨 核心改变：Y_target 变成差分值
        # Y_absolute: [y0, y1, ..., y29]
        y_abs = data_Y_absolute[i + input_len: i + input_len + output_len]
        # 差分计算：delta_y = y_t - y_{t-1}
        # 第一项的 y_{t-1} 是输入序列的最后一个值
        last_known_abs = data_Y_absolute[i + input_len - 1]
        # 构造包含前置项的序列，方便做差分
        y_abs_with_last = np.vstack([last_known_abs, y_abs])
        y_diff = np.diff(y_abs_with_last, axis=0)  # shape: (30, 1)
        if not (np.isnan(x_win).any() or np.isnan(y_diff).any()):
            X_list.append(x_win)
            Y_diff_list.append(y_diff)
            Y_last_abs_list.append(last_known_abs)  # 记录输入序列的最后绝对温度，用于还原
    return (np.array(X_list, dtype=np.float32),
            np.array(Y_diff_list, dtype=np.float32).squeeze(-1),
            np.array(Y_last_abs_list, dtype=np.float32))


# ==========================================
# 主流程
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("开始执行 Step 3 差分预测数据集构造")
    print("=" * 50)
    df = pd.read_parquet(Path(MERGED_DATA_DIR) / MERGED_FILENAME)
    df = add_momentum_features(df, TARGET_VAR)
    config_path = Path(CONFIG_DIR) / "feature_config.json"
    if config_path.exists():
        df = apply_lag_shifts(df, config_path, TARGET_VAR)
    df_train, df_val, df_demo = split_timeseries(df)
    features = [col for col in df.columns if col != TARGET_VAR]
    print("\n进行标准化...")
    df_train_norm, df_val_norm, df_demo_norm = normalize_and_save_params(df_train, df_val, df_demo,
                                                                         features + [TARGET_VAR])
    print("\n开始差分滑动窗口切分:")
    X_train, Y_train_diff, Y_train_last = create_diff_sliding_windows(df_train_norm, INPUT_LEN, OUTPUT_LEN, STRIDE,
                                                                      TARGET_VAR)
    X_val, Y_val_diff, Y_val_last = create_diff_sliding_windows(df_val_norm, INPUT_LEN, OUTPUT_LEN, STRIDE, TARGET_VAR)
    X_demo, Y_demo_diff, Y_demo_last = create_diff_sliding_windows(df_demo_norm, INPUT_LEN, OUTPUT_LEN, STRIDE,
                                                                   TARGET_VAR)
    print(f"  训练集 X: {X_train.shape}, Y_diff: {Y_train_diff.shape}")
    # 🚨 注意：我们要把 Y_last_abs 也存下来，评估时需要用它把差分还原成绝对温度
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"\n保存差分数据集至 {DATASET_DIR} ...")
    np.savez_compressed(Path(DATASET_DIR) / "train_data_diff.npz", X=X_train, Y_diff=Y_train_diff,
                        Y_last_abs=Y_train_last)
    np.savez_compressed(Path(DATASET_DIR) / "val_data_diff.npz", X=X_val, Y_diff=Y_val_diff, Y_last_abs=Y_val_last)
    np.savez_compressed(Path(DATASET_DIR) / "demo_data_diff.npz", X=X_demo, Y_diff=Y_demo_diff, Y_last_abs=Y_demo_last)
    print("\n🎉 Step 4 终极版执行完毕！模型即将学习'如何变化'而非'绝对数值'。")
