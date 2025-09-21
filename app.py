# import streamlit as st
# import torch
# from torchvision import transforms
# from PIL import Image
# import io

# # 导入我们之前编写的模型创建函数
# from src.model import create_model

# # --- Streamlit 缓存 ---
# # 这个装饰器@st.cache_resource能确保模型只在第一次加载时读取，
# # 后续用户操作时会直接复用内存中的模型，极大提高响应速度。
# @st.cache_resource
# # def load_model():
# #     """加载并返回训练好的模型"""
# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     model = create_model(num_classes=2)
    
# #     # 加载我们训练好的模型权重
# #     # 注意：确保模型文件路径相对于app.py是正确的
# #     model_path = 'saved_models/best_model.pth'
# #     model.load_state_dict(torch.load(model_path, map_location=device))
# #     model = model.to(device)
# #     model.eval() # 必须设置为评估模式
# #     return model, device

# @st.cache_resource
# def load_model():
#     device = torch.device("cpu") # 在共享平台，必须使用CPU
#     model = create_model(num_classes=2)         

#     # --- 请将这里的链接替换为您自己的模型下载链接 ---
#     model_url = "https://github.com/lang1210/pneumonia-Diagnasis-System/raw/refs/heads/main/saved_models/best_model5.pth?download=" 
#     model_path = 'saved_models/best_model5.pth'

#     # 如果模型文件在服务器上不存在，则从网上下载
#     if not os.path.exists(model_path):
#         # 显示下载提示
#         with st.spinner(f"正在从网络下载模型文件（约100MB），请稍候..."):
#             import requests
#             if not os.path.exists('saved_models'):
#                 os.makedirs('saved_models')

#             # 下载文件
#             r = requests.get(model_url, stream=True)
#             with open(model_path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)
#         st.success("模型下载完成！")

#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.to(device)
#     model.eval()
#     return model, device

# # --- 预测函数 ---
# def predict(model, image_bytes, device):
#     """对单张图片进行预测"""
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(outputs, dim=1)
#         confidence, predicted_idx = torch.max(probabilities, 1)

#     class_names = ['NORMAL', 'PNEUMONIA']
#     predicted_class = class_names[predicted_idx.item()]
    
#     return predicted_class, confidence.item()

# # --- 网页主函数 ---
# def main():
#     st.title("肺炎X光片智能诊断系统")
#     st.write("上传一张胸部X光片，模型将分析它是否为肺炎。")

#     # 加载模型
#     model, device = load_model()

#     # 创建一个文件上传组件
#     uploaded_file = st.file_uploader("请在此处上传图片...", type=["jpeg", "jpg", "png"])

#     if uploaded_file is not None:
#         # 读取上传的文件
#         image_bytes = uploaded_file.getvalue()
        
#         # 在网页上显示上传的图片
#         st.image(image_bytes, caption='已上传的X光片', use_container_width=True)
        
#         # 当用户点击“开始诊断”按钮时，执行预测
#         if st.button('开始诊断'):
#             with st.spinner('模型正在分析中，请稍候...'):
#                 predicted_class, confidence = predict(model, image_bytes, device)
            
#             st.success('诊断完成！')
#             st.metric(label="诊断结果", value=predicted_class)
#             st.metric(label="模型置信度", value=f"{confidence:.2%}")

#             if predicted_class == 'PNEUMONIA':
#                 st.warning("警告：诊断结果为肺炎，建议咨询专业医生。")
#             else:
#                 st.info("提示：诊断结果为正常。")

# if __name__ == '__main__':
#     main()

# app.py目标检测版本
import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import os
import io
import torchvision.transforms as T

# 导入我们为BDD100K项目编写的组件
from src.model1 import create_detection_model
from preprocess_bdd import BDD_CATEGORIES # 导入类别定义

# --- Streamlit 缓存 ---
@st.cache_resource
# app.py (替换旧的 load_model 函数)

# ... (确保文件顶部有 import os, import streamlit as st, import requests 等) ...

@st.cache_resource
def load_model():
    """加载模型，如果本地不存在，则从URL下载"""
    device = torch.device("cpu") # Streamlit Cloud是CPU环境
    model = create_detection_model(num_classes=11)
    
    # --- 请将这里的链接替换为您自己的模型直接下载链接 ---
    model_url = "https://github.com/lang1210/pneumonia-Diagnasis-System/raw/refs/heads/main/saved_models/best_detection_model.pth?download=" # 这是一个示例，请确保链接是您最新的、正确的模型
    model_path = 'saved_models/best_detection_model.pth'

    # 如果模型文件在服务器上不存在，则从网上下载
    if not os.path.exists(model_path):
        with st.spinner(f"首次加载，正在从网络下载模型文件（约100MB），请稍候..."):
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            
            try:
                import requests
                r = requests.get(model_url, stream=True)
                r.raise_for_status() # 如果下载链接有问题则会报错
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("模型下载完成！")
            except Exception as e:
                st.error(f"模型下载失败: {e}")
                return None, None
    
    # 从本地路径加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        st.error(f"加载模型权重失败: {e}")
        return None, None

    model = model.to(device)
    model.eval()
    return model, device
# --- 预测与绘图函数 ---
def predict_and_draw_boxes(model, image_bytes, device, confidence_threshold=0.5):
    """对单张图片进行预测，并返回画好框的图片"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose([T.ToTensor()]) 
    img_tensor = transform(img).to(device)
    
    with torch.no_grad():
        prediction = model([img_tensor])

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    idx_to_class = {v: k for k, v in BDD_CATEGORIES.items()}
    
    for i in range(len(prediction[0]['boxes'])):
        score = prediction[0]['scores'][i].item()
        
        if score > confidence_threshold:
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            label_idx = prediction[0]['labels'][i].item()
            label_name = idx_to_class.get(label_idx, 'Unknown')
            
            cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            text = f"{label_name}: {score:.2f}"
            cv2.putText(img_cv, text, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
    result_image_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return result_image_rgb

# --- 网页主函数 ---
def main():
    st.set_page_config(page_title="车辆目标检测系统", layout="wide")
    st.title("车辆目标检测智能系统")
    st.write("上传一张街景图片，模型将自动识别并框出其中的车辆、行人、交通标志等。")

    model, device = load_model()
    
    if model is not None:
        uploaded_file = st.file_uploader("请上传或拖拽您的街景图片到此处", type=["jpeg", "jpg", "png"])

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始图片")
                st.image(image_bytes, caption='已上传的图片', use_column_width=True)
            
            with col2:
                st.subheader("检测结果")
                if st.button('开始检测'):
                    with st.spinner('模型正在分析中，请稍候...'):
                        result_image = predict_and_draw_boxes(model, image_bytes, device)
                    
                    st.image(result_image, caption='模型检测结果', use_column_width=True)
                    st.success('检测完成！')

if __name__ == '__main__':
    main()