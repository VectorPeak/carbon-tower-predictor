# -*- coding: utf-8 -*-
"""
Step 4：差分预测模型训练
主要流程：
1. 读取差分数据集 (Y_diff)，训练模型拟合温度变化量。
2. 损失函数简化为 SmoothL1Loss (Huber Loss)，对极端温差更鲁棒。
3. 保留大容量网络 (Hidden=128, Layers=3) 和学习率自动衰减。
"""
import time

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 配置区
# ==========================================
DATASET_DIR = "data/datasets"
CONFIG_DIR = "configs"
MODEL_DIR = "artifacts/models"
INPUT_LEN = 30
OUTPUT_LEN = 30
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# ==========================================
# 模型定义 (大容量网络)
# ==========================================
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ==========================================
# 训练与验证逻辑
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    # 1. 加载数据 (注意文件名和字段名的改变)
    print("正在加载差分预测数据集...")
    train_data = np.load(Path(DATASET_DIR) / "train_data_diff.npz")
    val_data = np.load(Path(DATASET_DIR) / "val_data_diff.npz")
    X_train = torch.FloatTensor(train_data['X']).to(device)
    Y_train_diff = torch.FloatTensor(train_data['Y_diff']).to(device)  # 读取差分标签
    X_val = torch.FloatTensor(val_data['X']).to(device)
    Y_val_diff = torch.FloatTensor(val_data['Y_diff']).to(device)
    train_loader = DataLoader(TensorDataset(X_train, Y_train_diff), batch_size=BATCH_SIZE, shuffle=True)
    # 2. 初始化模型
    input_size = X_train.shape[2]
    model = EnhancedLSTM(input_size, hidden_size=128, num_layers=3, output_len=OUTPUT_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # 对差分标签使用 SmoothL1Loss
    # 相比MSE，SmoothL1Loss对极端的突变温差更鲁棒，不容易被异常值带偏
    criterion = nn.SmoothL1Loss()
    # 3. 训练循环
    best_val_loss = float('inf')
    print("\n开始训练差分预测模型...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for x_batch, y_diff_batch in train_loader:
            optimizer.zero_grad()
            pred_diff = model(x_batch)
            loss = criterion(pred_diff, y_diff_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred_diff = model(X_val)
            val_loss = criterion(val_pred_diff, Y_val_diff).item()
        # 更新学习率
        scheduler.step(val_loss)
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = Path(MODEL_DIR) / "best_model_v4_diff.pth"
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), Path(MODEL_DIR) / f"best_model_v4_diff_{time.time():.0f}.pth")
            tag = "验证集损失刷新最优"
        else:
            tag = ""
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']} {tag}")
    print("\n差分预测模型训练完成。")
    print("模型输出为'温度变化量'，评估时需要累加回绝对温度。")


if __name__ == "__main__":
    train_model()
