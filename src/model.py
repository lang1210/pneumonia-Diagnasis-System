# src/model.py

from torchvision import models
import torch.nn as nn

def create_model(num_classes=2):
    """
    创建一个预训练的ResNet-50模型，并替换最后的全连接层。
    
    Args:
        num_classes (int): 输出类别的数量 (我们是2分类：正常/肺炎)。
    
    Returns:
        model: 配置好的PyTorch模型。
    """
    # 加载一个在ImageNet上预训练好的ResNet-50模型
    model = models.resnet50(pretrained=True)
    
    # "冻结"模型的所有参数，这样在训练时它们就不会更新
    # 我们只训练我们自己添加的最后一层
    for param in model.parameters():
        param.requires_grad = False
        
    # 获取模型最后全连接层的输入特征数
    num_ftrs = model.fc.in_features
    
    # 替换掉原来的全连接层，换成我们自己的，输出维度为num_classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model