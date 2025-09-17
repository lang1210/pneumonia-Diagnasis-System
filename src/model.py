# src/model.py

# src/model.py

# from torchvision import models
# import torch.nn as nn
# # 在文件顶部，新增这一行导入
# from torchvision.models import ResNet50_Weights

# def create_model(num_classes=2):
#     # 获取当前最优的预训练权重
#     weights = ResNet50_Weights.DEFAULT
    
#     # 使用新的 'weights' 参数来替换 'pretrained=True'
#     model = models.resnet50(weights=weights)
    
#     # ... 函数的其余部分 ...
#     return model

# src/model.py




# 替换分类头为2，之前为默认resnet的1000
from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

def create_model(num_classes=2):
    """
    创建一个预训练的ResNet-50模型，并替换最后的全连接层。
    """
    # 获取当前最优的预训练权重
    weights = ResNet50_Weights.DEFAULT
    
    # 加载预训练模型
    model = models.resnet50(weights=weights)
    
    # 冻结所有原始参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 获取最后全连接层的输入特征数
    num_ftrs = model.fc.in_features
    
    # 替换掉原来的全连接层，换成我们自己的，输出维度为num_classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model