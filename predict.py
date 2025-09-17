import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

# 导入我们之前编写的模型创建函数
from src.model import create_model

def predict(model, image_path, device):
    """
    对单张图片进行预测。

    Args:
        model: 加载了权重的模型。
        image_path (str): 需要预测的图片的路径。
        device: 'cuda:0' 或 'cpu'。

    Returns:
        str: 预测的类别名称 ('NORMAL' 或 'PNEUMONIA')。
        float: 预测为该类别的置信度。
    """
    # 1. 准备图片预处理流程 (必须和验证集/测试集保持一致)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 加载并预处理图片
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"错误：文件不存在 {image_path}", 0.0
        
    image_tensor = transform(image).unsqueeze(0) # 增加一个批次维度 (batch dimension)
    image_tensor = image_tensor.to(device)

    # 3. 进行预测
    with torch.no_grad():
        outputs = model(outputs)
        # 使用 softmax 将输出转换为概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 获取概率最高的那一类的置信度和索引
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 4. 将预测的索引转换为类别名称
    class_names = ['NORMAL', 'PNEUMONIA'] # 确保这个顺序和训练时一致
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class, confidence.item()

if __name__ == '__main__':
    # --- 配置 ---
    MODEL_PATH = 'saved_models/best_model3.pth' # 训练好的模型路径
    
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description='对单张X光图片进行肺炎诊断')
    parser.add_argument('--image', type=str, required=True, help='需要预测的图片路径')
    args = parser.parse_args()

    # --- 准备工作 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 加载模型 ---
    model = create_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {MODEL_PATH}。请先运行 train.py 来训练并保存模型。")
        exit()
        
    model = model.to(device)
    model.eval() # 必须设置为评估模式

    # --- 执行预测并打印结果 ---
    predicted_class, confidence = predict(model, args.image, device)
    
    print("-" * 30)
    print(f"图片路径: {args.image}")
    print(f"诊断结果: {predicted_class}")
    print(f"置信度: {confidence:.4f}")
    print("-" * 30)