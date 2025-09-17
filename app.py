import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io

# 导入我们之前编写的模型创建函数
from src.model import create_model

# --- Streamlit 缓存 ---
# 这个装饰器@st.cache_resource能确保模型只在第一次加载时读取，
# 后续用户操作时会直接复用内存中的模型，极大提高响应速度。
@st.cache_resource
def load_model():
    """加载并返回训练好的模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2)
    
    # 加载我们训练好的模型权重
    # 注意：确保模型文件路径相对于app.py是正确的
    model_path = 'saved_models/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # 必须设置为评估模式
    return model, device

# --- 预测函数 ---
def predict(model, image_bytes, device):
    """对单张图片进行预测"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    class_names = ['NORMAL', 'PNEUMONIA']
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class, confidence.item()

# --- 网页主函数 ---
def main():
    st.title("肺炎X光片智能诊断系统")
    st.write("上传一张胸部X光片，模型将分析它是否为肺炎。")

    # 加载模型
    model, device = load_model()

    # 创建一个文件上传组件
    uploaded_file = st.file_uploader("请在此处上传图片...", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        # 读取上传的文件
        image_bytes = uploaded_file.getvalue()
        
        # 在网页上显示上传的图片
        st.image(image_bytes, caption='已上传的X光片', use_column_width=True)
        
        # 当用户点击“开始诊断”按钮时，执行预测
        if st.button('开始诊断'):
            with st.spinner('模型正在分析中，请稍候...'):
                predicted_class, confidence = predict(model, image_bytes, device)
            
            st.success('诊断完成！')
            st.metric(label="诊断结果", value=predicted_class)
            st.metric(label="模型置信度", value=f"{confidence:.2%}")

            if predicted_class == 'PNEUMONIA':
                st.warning("警告：诊断结果为肺炎，建议咨询专业医生。")
            else:
                st.info("提示：诊断结果为正常。")

if __name__ == '__main__':
    main()