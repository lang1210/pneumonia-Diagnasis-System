# # # src/data_loader.py

# # import torch
# # import os
# # from PIL import Image
# # import torchvision.transforms as T

# # # BDD100K 数据集中目标检测任务包含的10个类别
# # # 我们将它们映射为从1到10的整数。注意：0通常留给“背景”类别。
# # BDD_CATEGORIES = {
# #     "pedestrian": 1, "rider": 2, "car": 3, "truck": 4, "bus": 5,
# #     "train": 6, "motorcycle": 7, "bicycle": 8, "traffic light": 9, "traffic sign": 10
# # }

# # class Bdd100kDataset(torch.utils.data.Dataset):
# #     def __init__(self, image_dir, image_to_labels, transforms=None):
# #         """
# #         Args:
# #             image_dir (str): 包含所有图片的文件夹路径。
# #             image_to_labels (dict): 我们在Notebook中创建的，从图片名到标签列表的映射字典。
# #             transforms: 应用于图片的预处理和增强操作。
# #         """
# #         self.image_dir = image_dir
# #         self.image_to_labels = image_to_labels
# #         self.transforms = transforms
# #         # 我们只使用那些在标签文件里有记录的图片
# #         self.image_names = list(image_to_labels.keys())

# #     def __len__(self):
# #         # 数据集的总长度就是有标签的图片的数量
# #         return len(self.image_names)

# #     def __getitem__(self, idx):
# #         # 1. 根据索引获取图片名
# #         image_name = self.image_names[idx]
# #         image_path = os.path.join(self.image_dir, image_name)
        
# #         # 2. 读取图片
# #         image = Image.open(image_path).convert("RGB")
        
# #         # 3. 获取该图片的所有标注信息
# #         labels_info = self.image_to_labels[image_name]
        
# #         boxes = []
# #         labels = []
        
# #         # 4. 遍历这张图片上的每一个物体标注
# #         for label in labels_info:
# #             category = label.get('category')
# #             # 我们只关心那些在我们的类别映射表里存在的物体
# #             if category in BDD_CATEGORIES:
# #                 # 提取边界框坐标
# #                 box2d = label.get('box2d')
# #                 if box2d:
# #                     x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
# #                     boxes.append([x1, y1, x2, y2])
# #                     labels.append(BDD_CATEGORIES[category])

# #         # 5. 将边界框和标签转换为Tensor
# #         #    注意：必须是torch.as_tensor，而不是torch.tensor，以避免不必要的内存拷贝
# #         boxes = torch.as_tensor(boxes, dtype=torch.float32)
# #         labels = torch.as_tensor(labels, dtype=torch.int64)
        
# #         # 6. 创建一个字典来存放所有的目标信息
# #         #    这是Torchvision目标检测模型要求的标准格式
# #         target = {}
# #         target["boxes"] = boxes
# #         target["labels"] = labels
# #         target["image_id"] = torch.tensor([idx])
        
# #         # 7. 应用图像变换
# #         if self.transforms is not None:
# #             # 注意：在目标检测中，如果对图片做变换（如缩放），边界框也要做相应变换
# #             # 但为了简化第一步，我们只做最基础的ToTensor变换
# #             image = self.transforms(image)
            
# #         return image, target

# # def get_transform():
# #     """定义图像的预处理操作"""
# #     return T.Compose([
# #         T.ToTensor(),
# #     ])

# # def collate_fn(batch):
# #     """
# #     DataLoader的辅助函数，用来处理一个批次内图片和标签的打包。
# #     目标检测任务中必须使用，因为每张图片的物体数量不同。
# #     """
# #     return tuple(zip(*batch))

# # src/data_loader.py (最终版 - 带数据筛选功能)

# # 检查剔除空边界框
# # import torch
# # import os
# # from PIL import Image
# # import torchvision.transforms as T
# # from tqdm import tqdm # 引入进度条，方便查看筛选进度

# # # BDD100K 10个类别
# # BDD_CATEGORIES = {
# #     "pedestrian": 1, "rider": 2, "car": 3, "truck": 4, "bus": 5,
# #     "train": 6, "motorcycle": 7, "bicycle": 8, "traffic light": 9, "traffic sign": 10
# # }

# # class Bdd100kDataset(torch.utils.data.Dataset):
# #     def __init__(self, image_dir, image_to_labels, transforms=None):
# #         self.image_dir = image_dir
# #         self.transforms = transforms
        
# #         # --- 关键修正：在这里预先筛选出所有包含有效标注的图片 ---
# #         print("Filtering dataset for images with valid bounding box annotations...")
# #         self.valid_image_names = []
# #         all_image_names = list(image_to_labels.keys())
        
# #         for name in tqdm(all_image_names):
# #             labels_info = image_to_labels[name]
# #             has_valid_box = False
# #             for label in labels_info:
# #                 # 检查标签是否是包含'box2d'的目标检测标签，并且类别是我们关心的
# #                 if label.get('category') in BDD_CATEGORIES and label.get('box2d'):
# #                     has_valid_box = True
# #                     break # 只要找到一个有效的，就可以保留这张图了
            
# #             if has_valid_box:
# #                 self.valid_image_names.append(name)
                
# #         print(f"Filtering complete. Found {len(self.valid_image_names)} images with valid annotations.")
# #         # --- 修正结束 ---

# #     def __len__(self):
# #         # 数据集的总长度现在是有效图片的数量
# #         return len(self.valid_image_names)

# #     def __getitem__(self, idx):
# #         # 从“有效图片”列表中获取图片名
# #         image_name = self.valid_image_names[idx]
# #         image_path = os.path.join(self.image_dir, image_name)
# #         image = Image.open(image_path).convert("RGB")
        
# #         # 这里我们直接用 image_to_labels[image_name] 会有冗余，但为了代码清晰，暂时保留
# #         # 一个更优化的方法是在__init__时就把有效的labels也存起来
# #         labels_info = [label for label in image_to_labels[image_name] if label.get('category') in BDD_CATEGORIES and label.get('box2d')]
        
# #         boxes = []
# #         labels = []
        
# #         for label in labels_info:
# #             box2d = label['box2d']
# #             x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
# #             boxes.append([x1, y1, x2, y2])
# #             labels.append(BDD_CATEGORIES[label['category']])

# #         boxes = torch.as_tensor(boxes, dtype=torch.float32)
# #         labels = torch.as_tensor(labels, dtype=torch.int64)
        
# #         target = {}
# #         target["boxes"] = boxes
# #         target["labels"] = labels
# #         target["image_id"] = torch.tensor([idx])
        
# #         if self.transforms is not None:
# #             image = self.transforms(image)
            
# #         return image, target

# # def get_transform():
# #     return T.Compose([T.ToTensor()])

# # def collate_fn(batch):
# #     return tuple(zip(*batch))


# # src/data_loader.py (最终简化版)

import torch
import os
from PIL import Image
import torchvision.transforms as T

# 从预处理脚本中导入类别定义，保证一致性
from preprocess_bdd import BDD_CATEGORIES

class Bdd100kDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, image_to_labels, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        # 现在传入的image_to_labels字典已经是被完全筛选过的了
        self.image_names = list(image_to_labels.keys())
        self.image_to_labels = image_to_labels

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        # 因为数据已经预筛选，这里的逻辑变得非常直接
        labels_info = self.image_to_labels[image_name]
        
        boxes = []
        labels = []
        
        for label in labels_info:
            box2d = label['box2d']
            x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
            boxes.append([x1, y1, x2, y2])
            labels.append(BDD_CATEGORIES[label['category']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, target

def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))

# src/data_loader.py (带诊断打印的版本)

# import torch
# import os
# from PIL import Image
# import torchvision.transforms as T
# from tqdm import tqdm

# 从预处理脚本中导入类别定义，保证一致性
# 假设 preprocess_bdd.py 和 train.py 在同一个根目录
# from preprocess_bdd import BDD_CATEGORIES

# class Bdd100kDataset(torch.utils.data.Dataset):
#     def __init__(self, image_dir, image_to_labels, transforms=None):
#         self.image_dir = image_dir
#         self.transforms = transforms
#         self.image_names = list(image_to_labels.keys())
#         self.image_to_labels = image_to_labels
#         # 筛选步骤已在 preprocess_bdd.py 中完成

#     def __len__(self):
#         return len(self.image_names)

#     def __getitem__(self, idx):
#         # --- 诊断路标 1 ---
#         print(f"DataLoader: 正在尝试获取索引为 {idx} 的数据...")
        
#         image_name = self.image_names[idx]
#         image_path = os.path.join(self.image_dir, image_name)
        
#         # --- 诊断路标 2 ---
#         print(f"  -> 准备加载图片: {image_path}")
        
#         image = Image.open(image_path).convert("RGB")
        
#         # --- 诊断路标 3 ---
#         print(f"  -> 图片加载成功. 准备处理标签...")
        
#         labels_info = self.image_to_labels[image_name]
        
#         boxes = []
#         labels = []
        
#         for label in labels_info:
#             box2d = label['box2d']
#             x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
#             boxes.append([x1, y1, x2, y2])
#             labels.append(BDD_CATEGORIES[label['category']])

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)
        
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["image_id"] = torch.tensor([idx])
        
#         # --- 诊断路标 4 ---
#         print(f"  -> 标签处理完毕. 准备应用图像变换...")
        
#         if self.transforms is not None:
#             image = self.transforms(image)
        
#         # --- 诊断路标 5 ---
#         print(f"  -> 索引 {idx} 的数据已准备就绪. 即将返回.")
            
#         return image, target

# def get_transform():
#     return T.Compose([T.ToTensor()])

# def collate_fn(batch):
#     return tuple(zip(*batch))