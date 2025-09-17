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
    

    # src/data_loader.py
# 方案一：对“正常”样本进行更强的数据增强 (Class-Specific Augmentation)
# 思路：既然“正常”样本少，我们就对每一张“正常”样本图片做更多、更复杂的“化妆”（旋转、扭曲、变色等），让模型每次看到“正常”样本时，它都长得不太一样。这相当于人为地创造了更多、更多样化的“正常”样本，强迫模型去学习它们更本质的特征。

# 操作：这需要我们自定义一个Dataset类来替代ImageFolder。请用以下代码完整替换 src/data_loader.py 文件的内容
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 为“正常”类别定义更强力的数据增强
# 我们增加了随机旋转和颜色抖动
# strong_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),  # 随机旋转+/- 15度
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 颜色抖动
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # 为“肺炎”类别和验证集定义标准的数据增强/变换
# standard_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # 自定义一个数据集类，可以为不同类别应用不同变换
# class CustomImageFolder(Dataset):
#     def __init__(self, root, normal_transform=None, pneumonia_transform=None):
#         self.root = root
#         self.normal_transform = normal_transform
#         self.pneumonia_transform = pneumonia_transform

#         # 使用ImageFolder的内部函数来获取图片路径和标签
#         self.dataset = datasets.ImageFolder(root)
#         self.classes = self.dataset.classes
#         self.class_to_idx = self.dataset.class_to_idx
#         self.samples = self.dataset.samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, target = self.samples[idx]
#         sample = Image.open(path).convert("RGB")

#         # 根据标签（0代表NORMAL, 1代表PNEUMONIA）应用不同的变换
#         if target == self.class_to_idx['NORMAL']:
#             if self.normal_transform:
#                 sample = self.normal_transform(sample)
#         else: # PNEUMONIA
#             if self.pneumonia_transform:
#                 sample = self.pneumonia_transform(sample)

#         return sample, target

# # create_dataloaders 函数需要重写
# def create_dataloaders(data_dir, batch_size):
#     # 训练集使用我们自定义的数据集类
#     train_dataset = CustomImageFolder(
#         root=os.path.join(data_dir, 'train'),
#         normal_transform=strong_transforms,
#         pneumonia_transform=transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     )

#     # 验证集依然使用标准的ImageFolder和标准变换
#     val_dataset = datasets.ImageFolder(
#         os.path.join(data_dir, 'val'),
#         standard_transforms
#     )

#     dataloaders = {
#         'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
#         'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     }

#     dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

#     return dataloaders, dataset_sizes