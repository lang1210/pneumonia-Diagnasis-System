# pneumonia-Diagnasis-System
 deep learning project to classify pneumonia from chest X-ray images, with model quantization and a Streamlit web app
1.
   # 使用resnet50+交叉熵损失
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
 # --- 修改损失函数 ---
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
混淆矩阵 (Confusion Matrix):

[[151  83]

 [  9 381]]

用了方案一。貌似没有任何提升
------------------------------
3
# 对“正常”样本进行更强的数据增强 (Class-Specific Augmentation)
思路：既然“正常”样本少，我们就对每一张“正常”样本图片做更多、更复杂的“化妆”（旋转、扭曲、变色等），让模型每次看到“正常”样本时，它都长得不太一样。这相当于人为地创造了更多、更多样化的“正常”样本，强迫模型去学习它们更本质的特征。
分类报告 (Classification Report):
              precision    recall  f1-score   support

      NORMAL     0.9115    0.7479    0.8216       234
   PNEUMONIA     0.8634    0.9564    0.9075       390

    accuracy                         0.8782       624
   macro avg     0.8874    0.8521    0.8646       624
weighted avg     0.8814    0.8782    0.8753       624

混淆矩阵 (Confusion Matrix):
[[151  83]
 [  9 381]]

 [[175  59]
 [ 17 373]]

混淆矩阵图片已保存为 confusion_matrix.png
