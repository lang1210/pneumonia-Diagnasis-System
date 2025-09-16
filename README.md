# pneumonia-Diagnasis-System
 deep learning project to classify pneumonia from chest X-ray images, with model quantization and a Streamlit web app
1. 使用resnet50+交叉熵损失
分类报告 (Classification Report):

                precision   recall  f1-score   support



      NORMAL     0.9430    0.6368    0.7602       234

   PNEUMONIA     0.8176    0.9769    0.8902       390



    accuracy                         0.8494       624

   macro avg     0.8803    0.8068    0.8252       624

weighted avg     0.8646    0.8494    0.8414       624



------------------------------

混淆矩阵 (Confusion Matrix):

[[149  85]

 [  9 381]]
