# pneumonia-Diagnasis-System
 deep learning project to classify pneumonia from chest X-ray images, with model quantization and a Streamlit web app
肺炎X光片智能诊断系统 (Pneumonia Diagnosis System)
这是一个基于PyTorch和Streamlit构建的端到端深度学习项目，旨在通过胸部X光片图像，智能地辅助诊断是否患有肺炎。项目不仅涵盖了从数据处理、模型训练、性能评估的全过程，还特别关注了真实世界中常见的数据不均衡和域外样本问题，并最终将模型封装成了一个公开、可交互的Web应用。

🚀 在线应用演示 (Live Demo)
您可以通过以下链接，直接访问并使用这个智能诊断系统：

➡️ https://your-app-name.streamlit.app/ ⬅️

✨ 主要功能 (Features)
高精度二分类：能有效地区分“正常(NORMAL)”与“肺炎(PNEUMONIA)”的胸部X光片。

鲁棒性设计：通过引入“无效(INVALID)”类别，模型能够识别并拒绝处理非X光片的无关图像，避免在真实场景中做出无意义的错误判断。

交互式Web界面：用户可以通过简洁的网页界面，轻松上传X光片图片，并即时获取模型的诊断结果和置信度。

模型自动下载：应用在云端首次启动时，会自动从指定的URL下载训练好的模型权重，实现了代码与模型的解耦。

🛠️ 技术栈 (Tech Stack)
语言: Python 3.9

核心框架: PyTorch, Torchvision

数据科学库: Scikit-learn, Pandas, Matplotlib, Seaborn

Web应用与部署: Streamlit, Streamlit Community Cloud

版本控制与文件管理: Git, Git LFS

🔧 如何在本地运行？ (Setup and Usage)
克隆仓库

Bash

git clone https://github.com/lang1210/pneumonia-Diagnosis-System.git
cd pneumonia-Diagnosis-System
创建并激活Conda环境

Bash

conda env create -f environment.yml
conda activate torch_env
下载模型文件
本项目使用Git LFS管理模型文件。在克隆仓库后，模型文件 saved_models/best_model.pth 应该已自动下载。

如果模型文件下载失败，您也可以从以下链接手动下载，并将其放置在 saved_models/ 目录下：
点击此处下载模型 (best_model.pth)
4.  运行关键脚本

训练模型: python train.py

评估模型: python evaluate.py

启动Web应用: streamlit run app.py

🎯 项目挑战与解决方案 (Key Challenges & Solutions)
1. 训练数据严重不均衡
问题: 训练集中“肺炎”样本数量远超“正常”样本，导致模型严重偏向于将结果预测为“肺炎”，使得对“正常”样本的召回率极低（仅63.7%）。

解决方案:

尝试1 (加权损失函数): 通过在CrossEntropyLoss中设置weight参数，对少数类的错判施加更大惩罚。结果有微小提升，但不足以扭转偏见。

尝试2 (针对性强数据增强): 通过编写自定义Dataset类，仅对“正常”这一个少数类样本，在数据加载时施加更强的数据增强（如随机旋转、颜色抖动），最终将“正常”类别的召回率显著提升至74.8%。

尝试3 (加权随机采样): 通过WeightedRandomSampler在每个批次中过采样少数类，也取得了与方法2相似的良好效果，验证了解决问题的多种途径。

2. 模型对无关图片的鲁棒性差
问题: 初始模型在输入一张风景照时，会以高置信度将其误判为“肺炎”，这在实际应用中是不可接受的。

解决方案: 引入“域外样本”概念，从公开数据集中搜集了约1500张无关图片，构成第三个“无效”类别。将模型从二分类重构为三分类进行训练，使其学会了有效识别并拒绝处理非X光片输入。

3. 复杂的服务器环境与部署难题
问题: 在部署过程中，遇到了目标服务器底层库(GLIBCXX)版本过低、无法兼容新版PyTorch/NumPy等一系列棘手的环境依赖问题。此外，服务器网络受限，也导致VS Code远程开发工具链自动安装失败。

解决方案:

环境层面: 通过系统性排查，最终确定了一套与旧版系统兼容的Python及核心库的稳定版本组合，并重建了纯净的Conda环境，彻底解决了依赖冲突。

开发工具层面: 在VS Code Remote-SSH连接受阻时，果断切换开发模式，采用VS Code + SSH FS的方案，实现了在功能受限环境下的高效远程开发。

部署层面: 解决了GitHub对大文件的限制（通过Git LFS）和Streamlit Cloud对依赖文件格式（requirements.txt的pip格式）、Python版本的要求，最终成功将应用部署到公网。

📊 最终结果 (Results)
经过多轮优化，最终的三分类模型在测试集上表现稳健，关键的混淆矩阵如下：


1.
 ### 1. 基线模型（二分类）的表现

该模型未经过针对性优化，少数类（NORMAL）的召回率较低，证明了数据不均衡带来的负面影响。

| 类别 (Class) | 精确率 (Precision) | 召回率 (Recall) | F1-Score | support |
|:----------:|:------------------:|:---------------:|:----------:| 
| **NORMAL** |        0.9430        |      0.6368       |    0.7602    | 234 |
| **PNEUMONIA**|        0.8176        |      0.9769       |    0.8902    |  390 |
| **总准确率 (Overall Accuracy)** |         -          |       -        |  0.8494 |
### 分类报告 (Classification Report)

| 类别 (Class) | 精确率 (Precision) | 召回率 (Recall) | F1-Score | support |

|:---：|:---:|:---:|:---:|:---:|

| **NORMAL** | 0.9430 | 0.6368 | 0.7602 | 234 |

| **PNEUMONIA** | 0.8176 | 0.9769 | 0.8902 | 390 |

| **总准确率 (Accuracy)** | - | - | 0.8494 | 624 |

### 使用resnet50+交叉熵损失
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
4
# 加权随机采样 (Weighted Random Sampling)
 思路：在每个训练批次中，强行让模型多看一些“正常”样本。比如正常情况下抽10张图，可能有7张肺炎3张正常；加权后，我们可以让它每次都抽5张肺炎5张正常，实现批次内的数据均衡
 
 分类标签: ['NORMAL', 'PNEUMONIA']
------------------------------
分类报告 (Classification Report):
              precision    recall  f1-score   support

      NORMAL     0.9105    0.7393    0.8160       234
   PNEUMONIA     0.8594    0.9564    0.9053       390

    accuracy                         0.8750       624
   macro avg     0.8850    0.8479    0.8607       624
weighted avg     0.8786    0.8750    0.8719       624

------------------------------
混淆矩阵 (Confusion Matrix):
[[173  61]
 [ 17 373]]
5
# 换全连接层分类头为2
------------------------------
分类报告 (Classification Report):
              precision    recall  f1-score   support

      NORMAL     0.9290    0.7265    0.8153       234
   PNEUMONIA     0.8549    0.9667    0.9073       390

    accuracy                         0.8766       624
   macro avg     0.8919    0.8466    0.8613       624
weighted avg     0.8827    0.8766    0.8728       624

------------------------------
混淆矩阵 (Confusion Matrix):
[[170  64]
 [ 13 377]]

混淆矩阵图片已保存为 confusion_matrix.png
