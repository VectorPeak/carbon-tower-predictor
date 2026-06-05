# 📘 step1_merge_16_vars 教学文档：

# 工业多源 CSV 数据清洗与宽表合并

## 一、这个文件是做什么的？

想象你是一个工厂的数据工程师。现场有 13 台传感器，每台传感器每分钟都在往电脑里存一个 `.csv` 文件，记录"温度""压力""流量"等数据。这些文件格式混乱，有的带质量戳（比如用 `192` 表示"数据无效"），有的列名不统一。

**这个脚本的目标就是：**

1. 把这 13 个文件全部读进来
2. 把"时间"和"数值"自动找出来
3. 把"垃圾数据"（质量戳）扔掉
4. 按时间对齐，拼成一张"大宽表"（每一行是一个时间点，每一列是一个传感器）
5. 统一重采样到"每 1 分钟一条"，存成高性能文件

------

## 二、整体流程图（文字版）




```plain
13个CSV文件
    ↓
逐个读取：找时间列、找数值列、扔掉垃圾值
    ↓
13个独立的小表格（每个只有2列：时间 + 一个变量）
    ↓
按时间轴"横向拼接"成宽表（13列 + 1个时间索引）
    ↓
处理缺失值（前向/后向填充）
    ↓
重采样到1分钟均值
    ↓
保存为 merged_wide_table.parquet
```

------

## 三、代码分段拆解

### 3.1 头部：导入工具包




```python
import pandas as pd
import os
from pathlib import Path
```

------

### 3.2 配置区：告诉程序"我要处理哪些文件"


```python
RAW_DATA_DIR = "./raw_data/"
MERGED_DATA_DIR = "./merged_data/"
TARGET_VAR = "碳化取出液温度"

VARS_CONFIG = {
    "A-1CO2气进气压力": "CO2进气压力",
    "A-1出口尾气压力": "出口尾气压力",
    # ... 省略中间 ...
    "A-1碳化取出液温度": TARGET_VAR,
}
```

**字典的使用方式举例：**


```python
# 遍历字典：每次拿到一个"关键词"和对应的"短名字"
for keyword, col_name in VARS_CONFIG.items():
    print(f"我要找文件名包含 {keyword} 的文件，合并后叫 {col_name}")
```

------

### 3.3 核心函数 1：`read_single_var_csv`

这个函数负责**读一个 CSV 文件，并把它整理成干净的 2 列表格**（时间 + 数值）。


```python
def read_single_var_csv(file_path, col_name):
    """智能识别包含 datavalue 的列"""
    # 先读取表头
    df_temp = pd.read_csv(file_path)
```

------

#### 3.3.1 智能找列：不用硬编码列名


```python
    value_cols = [c for c in df_temp.columns if 'value' in c.lower()]
    time_cols = [c for c in df_temp.columns if 'time' in c.lower() or 'date' in c.lower()]
```

**为什么这样做？** 因为不同传感器厂家导出的 CSV，列名可能叫 `datavalue`、`value`、`pv`（process value）等。如果写死 `df['datavalue']`，一旦文件名换了就报错。**用"关键词模糊匹配"更鲁棒**。

------

#### 3.3.2 异常处理：找不到列就报错，别瞎猜


```python
    if not value_cols:
        raise ValueError(f"文件 {file_path.name} 中未找到包含 'value' 的列名！")
    if not time_cols:
        raise ValueError(f"文件 ... 中未找到包含 'time' 或 'date' 的列名！")
```

------

#### 3.3.3 提取并整理成标准 2 列表


```python
    time_col = time_cols[0]
    value_col = value_cols[0]
    
    df_selected = df_temp[[time_col, value_col]].copy()
    df_selected.columns = ['datetime', col_name]
```

------

#### 3.3.4 时间处理：把字符串变成"真时间"




```python
    df_selected['datetime'] = pd.to_datetime(df_selected['datetime'])
    df_selected = df_selected.set_index('datetime')
```

------

#### 3.3.5 数据清洗：干掉质量戳，填上缺失值


```python
    df_selected[col_name] = pd.to_numeric(df_selected[col_name], errors='coerce')
    df_selected = df_selected.dropna().ffill().bfill()
```

**这是整个 Step 1 最关键、最容易被忽视的三行代码。**

**为什么工业数据要 ffill + bfill？** 因为传感器偶尔丢包 1~2 分钟，温度不可能突变。用前后的真实值填充，比直接删行或填 0 更合理。

------

### 3.4 核心函数 2：`merge_all_data`

这个函数负责把 13 个小表格，按时间轴拼成一张"大宽表"。


```python
def merge_all_data(raw_dir, vars_config):
    all_dfs = []
    print("开始读取并智能提取 datavalue ...")
    for keyword, col_name in vars_config.items():
        # ... 找文件、读取、加入 all_dfs ...
```

------

#### 3.4.1 横向拼接：时间对齐




```python
    df_wide = pd.concat(all_dfs, axis=1, join='outer')
    df_wide = df_wide.ffill().bfill()
    df_wide = df_wide.resample('1min').mean().ffill()
    df_wide = df_wide.dropna()
```

**这是整个 Step 1 的核心，四行代码做四件大事。**

**为什么重采样用 `mean()`？** 因为 1 分钟内温度/压力可能波动 3~5 次，取均值比取最后一个值更能代表这一分钟的真实状态，也更有抗噪性。

------

### 3.5 主程序入口




```python
if __name__ == "__main__":
    print("=" * 50)
    print("开始执行 Step 1 : 提取 datavalue 宽表合并")
    print("=" * 50)
    df_final = merge_all_data(RAW_DATA_DIR, VARS_CONFIG)
    if df_final is not None:
        os.makedirs(MERGED_DATA_DIR, exist_ok=True)
        save_path = Path(MERGED_DATA_DIR) / "merged_wide_table.parquet"
        df_final.to_parquet(save_path)
        print(f"\n宽表已保存至: {save_path}")
        print("\n数据概况 (验证不再是192)：")
        print(df_final.describe().T[['mean', 'std', 'min', 'max']])
```

**生成的宽表**

```bash
数据形状: (43200, 13)
列名: ['CO2进气压力', '出口尾气压力', '反应段中部温度', '进塔氨母液II流量阀位', '进塔氨母液II流量', '进塔碳氨母液II流量阀位', '进塔碳氨母液II流量', '水箱冷却水出口温度', '冷却水压力', '碳化取出液温度阀位', '取出液流量', '液位', '碳化取出液温度']

前5行预览:
                            CO2进气压力    出口尾气压力  ...         液位    碳化取出液温度
datetime                                       ...                      
2026-04-23 00:00:00+08:00  0.323236  0.069691  ...  74.505362  41.284848
2026-04-23 00:01:00+08:00  0.322783  0.069376  ...  72.905255  41.332263
2026-04-23 00:02:00+08:00  0.322332  0.069225  ...  71.391116  41.415497
2026-04-23 00:03:00+08:00  0.322786  0.070238  ...  70.234410  41.497974
2026-04-23 00:04:00+08:00  0.323714  0.072059  ...  69.355181  41.553657

[5 rows x 13 columns]
```



------
