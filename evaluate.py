import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 导入我们之前编写的模型创建函数
from src.model import create_model

# --- 配置 ---
# 服务器上的数据集路径
DATA_DIR = 'data/ChestXRay2017/chest_ray/' 
# 我们训练好的、最佳模型的路径
MODEL_PATH = 'saved_models/best_model.pth'
BATCH_SIZE = 32

# --- 准备工作 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 数据加载 ---
# 注意：对于测试集，我们只使用和验证集一样的变换，不做随机数据增强
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), data_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
class_names = test_dataset.classes
print("测试集加载完成...")
print(f"分类标签: {class_names}")


# --- 加载模型 ---
model = create_model(num_classes=2)
# 加载我们训练好的模型权重
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
# 将模型设置为评估模式
model.eval()
print("模型加载完成，并设置为评估模式。")

# --- 开始评估 ---
all_preds = []
all_labels = []

# 在评估时，我们不需要计算梯度
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("评估完成！正在生成报告...")
print("-" * 30)

# --- 生成并打印评估报告 ---
# 1. 分类报告 (Classification Report)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("分类报告 (Classification Report):")
print(report)
print("-" * 30)

# 2. 混淆矩阵 (Confusion Matrix)
conf_matrix = confusion_matrix(all_labels, all_preds)
print("混淆矩阵 (Confusion Matrix):")
print(conf_matrix)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# 保存混淆矩阵图片
plt.savefig('confusion_matrix.png')
print("\n混淆矩阵图片已保存为 confusion_matrix.png")