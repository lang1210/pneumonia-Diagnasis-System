# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from src.data_loader import create_dataloaders
from src.model import create_model

# --- 1. 定义超参数和设置 ---
DATA_DIR = '../data/ChestXRay2017/chest_xray/' # 您的数据集路径
BATCH_SIZE = 32
NUM_EPOCHS = 10 # 先训练10个周期看看效果
LEARNING_RATE = 0.001

# 自动选择设备 (GPU或CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. 加载数据 ---
dataloaders, dataset_sizes = create_dataloaders(DATA_DIR, BATCH_SIZE)
print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['val']}")

# --- 3. 创建模型、损失函数和优化器 ---
model = create_model(num_classes=2)
model = model.to(device) # 将模型移动到指定设备

# 定义损失函数 (交叉熵损失)
#criterion = nn.CrossEntropyLoss()                                      交叉熵损失
# --- 修改损失函数 ---
# 根据我们训练集的样本数 (NORMAL=1349, PNEUMONIA=3884) 计算权重
# 权重计算逻辑：总样本数 / (类别数 * 该类别样本数)
# 目的是让样本数少的类别获得更高的权重
weights = torch.tensor([ (1349+3884)/(2*1349), (1349+3884)/(2*3884) ]).to(device)

# 将权重传入损失函数
criterion = nn.CrossEntropyLoss(weight=weights)                         #加权重损失
# 定义优化器 (只优化我们新加的那一层参数)
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# --- 4. 训练循环 ---
since = time.time()
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
    print('-' * 10)

    # 每个epoch都有一个训练和验证阶段
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 设置模型为训练模式
        else:
            model.eval()   # 设置模型为评估模式

        running_loss = 0.0
        running_corrects = 0

        # 遍历数据
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 只在训练阶段进行反向传播和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 保存表现最好的模型
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            # 确保saved_models文件夹存在
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print("Best model saved!")

    print()

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Acc: {best_acc:4f}')
#test