# pneumonia-Diagnasis-System
 deep learning project to classify pneumonia from chest X-ray images, with model quantization and a Streamlit web app
1.
   使用resnet50+交叉熵损失
   criterion = nn.CrossEntropyLoss()
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


2.
 --- 修改损失函数 ---
 根据我们训练集的样本数 (NORMAL=1349, PNEUMONIA=3884) 计算权重
 权重计算逻辑：总样本数 / (类别数 * 该类别样本数)
 目的是让样本数少的类别获得更高的权重
weights = torch.tensor([ (1349+3884)/(2*1349), (1349+3884)/(2*3884) ]).to(device)

将权重传入损失函数
criterion = nn.CrossEntropyLoss(weight=weights)
分类报告 (Classification Report):
              precision    recall  f1-score   support

      NORMAL     0.9437    0.6453    0.7665       234
   PNEUMONIA     0.8211    0.9769    0.8923       390

    accuracy                         0.8526       624
   macro avg     0.8824    0.8111    0.8294       624
weighted avg     0.8671    0.8526    0.8451       624

------------------------------
混淆矩阵 (Confusion Matrix):
[[151  83]
 [  9 381]]
