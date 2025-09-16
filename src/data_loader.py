# src/data_loader.py

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义图像预处理的步骤
# 这是非常关键的一步，尤其是Normalize的参数，是基于ImageNet数据集的统计得出的标准值
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), # 缩放到256x256
        transforms.CenterCrop(224), # 中心裁剪到224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def create_dataloaders(data_dir, batch_size):
    """
    创建一个函数来封装数据加载的过程，方便在主脚本中调用。
    
    Args:
        data_dir (str): 数据集的根目录 (应包含 train 和 val 文件夹)。
        batch_size (int): 每个批次加载的图片数量。

    Returns:
        dataloaders (dict): 包含训练和验证数据加载器的字典。
        dataset_sizes (dict): 包含训练和验证数据集大小的字典。
    """
    # 使用 ImageFolder 来创建数据集，它会自动从文件夹名中读取标签
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    # 创建数据加载器
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    return dataloaders, dataset_sizes