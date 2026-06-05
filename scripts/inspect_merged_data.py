import pandas as pd

df = pd.read_parquet("data/merged/merged_wide_table.parquet")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("\n前5行预览:")
print(df.head())
