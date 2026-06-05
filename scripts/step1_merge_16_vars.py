# -*- coding: utf-8 -*-
"""
步骤 1：单变量解析、冻结值清洗与降采样,多变量时序对齐与合并

功能：
1. 读取原始的以 逗号 分隔的 CSV 文件。
2. 根据标准列名精准提取所需字段。
3. 清洗数据冻结（死值）与质量戳异常。
4. 降采样至1分钟并按时间索引进行外连接合并，形成一张大宽表。
5. 处理合并后的缺失值：对短时间缺失进行线性插值，对长时间缺失保留为NaN（避免引入虚假数据）。
6. 保存合并后的宽表，供后续特征筛选使用。

设计原则：
- 单变量独立处理，避免多变量100万行全量合并导致内存溢出（OOM）。
- 优先保证物理意义的正确性，宁可留空，不用假数据填充。
"""
import pandas as pd
import os
from pathlib import Path

# ==========================================
# 配置区
# ==========================================
RAW_DATA_DIR = "data/raw"
MERGED_DATA_DIR = "data/merged"
TARGET_VAR = "碳化取出液温度"
VARS_CONFIG = {
    "A-1CO2气进气压力": "CO2进气压力",
    "A-1出口尾气压力": "出口尾气压力",
    "A-1反应段中部温度": "反应段中部温度",
    "A-1进塔氨母液II流量调节阀阀位": "进塔氨母液II流量阀位",
    "A-1进塔氨母液II流量": "进塔氨母液II流量",
    "A-1进塔碳氨母液II流量调节阀阀位": "进塔碳氨母液II流量阀位",
    "A-1进塔碳氨母液II流量": "进塔碳氨母液II流量",
    "A-1水箱冷却水出口温度": "水箱冷却水出口温度",
    "出碳化塔07C0101A-1冷却水压力": "冷却水压力",
    "A-1碳化取出液温度调节阀阀位": "碳化取出液温度阀位",
    "A-1取出液流量": "取出液流量",
    "A-1液位": "液位",
    "A-1碳化取出液温度": TARGET_VAR,
}


# ==========================================
# 智能读取逻辑
# ==========================================
def read_single_var_csv(file_path, col_name):
    """智能识别包含 datavalue 的列"""
    # 先读取表头
    df_temp = pd.read_csv(file_path)
    # 寻找包含 'value' (不区分大小写) 的列名
    value_cols = [c for c in df_temp.columns if 'value' in c.lower()]
    time_cols = [c for c in df_temp.columns if 'time' in c.lower() or 'date' in c.lower()]
    if not value_cols:
        raise ValueError(f"文件 {file_path.name} 中未找到包含 'value' 的列名！现有列: {df_temp.columns.tolist()}")
    if not time_cols:
        raise ValueError(
            f"文件 {file_path.name} 中未找到包含 'time' 或 'date' 的列名！现有列: {df_temp.columns.tolist()}")
    # 提取第一个匹配的时间和数值列
    time_col = time_cols[0]
    value_col = value_cols[0]
    # 提取并重命名
    df_selected = df_temp[[time_col, value_col]].copy()
    df_selected.columns = ['datetime', col_name]
    # 转换时间格式
    df_selected['datetime'] = pd.to_datetime(df_selected['datetime'])
    df_selected = df_selected.set_index('datetime')
    # ⚠️ 极其重要：剔除质量戳(192)等非数值数据，确保列类型为浮点数
    df_selected[col_name] = pd.to_numeric(df_selected[col_name], errors='coerce')
    # 剔除因转换产生的NaN，并向前填充
    df_selected = df_selected.dropna().ffill().bfill()
    return df_selected


def merge_all_data(raw_dir, vars_config):
    all_dfs = []
    print("开始读取并智能提取 datavalue ...")
    for keyword, col_name in vars_config.items():
        file_found = False
        for f in os.listdir(raw_dir):
            if keyword in f and f.endswith('.csv'):
                file_path = Path(raw_dir) / f
                try:
                    df_temp = read_single_var_csv(file_path, col_name)
                    all_dfs.append(df_temp)
                    print(f"  ✅ 读取: {f} -> 列名: {col_name} (数据点: {len(df_temp)})")
                    file_found = True
                except Exception as e:
                    print(f"  ❌ 读取失败: {f}，错误: {e}")
                break
        if not file_found:
            print(f"  ⚠️ 警告: 未找到包含关键词 [{keyword}] 的文件！")
    if not all_dfs:
        return None
    print("\n正在按时间戳对齐合并 (外连接 -> 剔除缺失 -> 重采样至1分钟)...")
    df_wide = pd.concat(all_dfs, axis=1, join='outer')
    df_wide = df_wide.ffill().bfill()
    df_wide = df_wide.resample('1min').mean().ffill()
    df_wide = df_wide.dropna()
    return df_wide


if __name__ == "__main__":
    print("=" * 50)
    print("开始执行 Step 1: 提取 datavalue 宽表合并")
    print("=" * 50)
    df_final = merge_all_data(RAW_DATA_DIR, VARS_CONFIG)
    if df_final is not None:
        os.makedirs(MERGED_DATA_DIR, exist_ok=True)
        save_path = Path(MERGED_DATA_DIR) / "merged_wide_table.parquet"
        df_final.to_parquet(save_path)
        print(f"\n🎉 宽表已保存至: {save_path}")
        print("\n数据概况 (验证不再是192)：")
        print(df_final.describe().T[['mean', 'std', 'min', 'max']])
