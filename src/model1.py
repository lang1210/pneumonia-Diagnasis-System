# # src/model.py

# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # src/model.py

# from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights # <--- 新增导入
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def create_detection_model(num_classes):
#     """
#     创建一个预训练的Faster R-CNN模型，并替换头部分类器。
#     """
#     # --- 修改部分开始 ---
#     # 使用新的、推荐的 weights 参数来加载预训练权重
#     weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
#     model = fasterrcnn_resnet50_fpn(weights=weights)
#     # --- 修改部分结束 ---

#     # 获取分类器的输入特征数
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
    
#     # 用一个新的头部替换掉预训练的头部
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
#     return model


from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_detection_model(num_classes):
    """
    创建一个预训练的Faster R-CNN模型，并替换头部分类器。
    使用最新的 weights API 来保证行为一致性。
    """
    # 使用最新的、官方推荐的 weights 参数加载预训练权重
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 用一个新的头部替换掉预训练的头部，以匹配我们自己的类别数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model