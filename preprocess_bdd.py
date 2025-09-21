# # preprocess_bdd.py

# import os
# import ijson
# from tqdm import tqdm
# import pickle

# # preprocess_bdd.py

# # ... (import语句不变) ...

# def process_and_save(data_type='train'):
#     """
#     读取BDD100K的原始JSON标签文件，处理后保存为pickle文件。
#     """
#     print(f"--- 开始处理 {data_type} 数据 ---")
    
#     # --- 修正后的路径定义 ---
#     bdd_root = '/home/teacherz/data/wangzhen'
    
#     # 我们直接使用 data_type 变量，而不是带花括号的字符串
#     label_filename = f'bdd100k_labels_images_{data_type}.json'
#     pickle_filename = f'image_to_labels_{data_type}.pkl'

#     label_path = os.path.join(bdd_root, data_type, 'annotations', label_filename)
#     pickle_path = os.path.join(bdd_root, data_type, 'annotations', pickle_filename)
#     # --- 修正结束 ---

#     # 检查原始JSON文件是否存在
#     if not os.path.exists(label_path):
#         # 同样修正这里的打印语句，使用 f-string 来正确显示路径
#         print(f"错误：找不到原始标签文件 {label_path}")
#         return
        
#     # ... (后续代码不变) ...

#     print(f"正在从 '{label_path}' 解析...")
    
#     image_to_labels = {}
#     total_items = 70000 if data_type == 'train' else 10000 # 为进度条设置大概的总数

#     try:
#         with open(label_path, 'r') as f:
#             parser = ijson.items(f, 'item')
#             for item in tqdm(parser, total=total_items):
#                 image_to_labels[item['name']] = item['labels']
#     except Exception as e:
#         print(f"解析JSON时出错: {e}")
#         return

#     print(f"解析完成！正在将 {len(image_to_labels)} 条数据保存到 '{pickle_path}' ...")
    
#     # 将处理好的字典保存为pickle文件
#     with open(pickle_path, 'wb') as f:
#         pickle.dump(image_to_labels, f)
        
#     print(f"成功保存！--- {data_type} 数据处理完毕 ---\n")

# if __name__ == '__main__':
#     # 依次处理训练集和验证集
#     process_and_save('train')
#     process_and_save('val')
#     print("所有数据预处理完成！")

# preprocess_bdd.py (最终优化版)
# 在这里提前清洗好了标签
import os
import ijson
from tqdm import tqdm
import pickle

# 将类别定义放在这里，方便data_loader导入和使用
BDD_CATEGORIES = {
    "pedestrian": 1, "rider": 2, "car": 3, "truck": 4, "bus": 5,
    "train": 6, "motorcycle": 7, "bicycle": 8, "traffic light": 9, "traffic sign": 10
}

def process_and_save(data_type='train'):
    print(f"--- 开始处理并筛选 {data_type} 数据 ---")
    
    bdd_root = '/home/teacherz/data/wangzhen'
    label_filename = f'bdd100k_labels_images_{data_type}.json'
    pickle_filename = f'image_to_labels_{data_type}.pkl'
    label_path = os.path.join(bdd_root, data_type, 'annotations', label_filename)
    pickle_path = os.path.join(bdd_root, data_type, 'annotations', pickle_filename)
    
    if not os.path.exists(label_path):
        print(f"错误：找不到原始标签文件 {label_path}")
        return

    print(f"正在从 '{label_path}' 解析并筛选...")
    
    image_to_labels_filtered = {}
    total_items = 70000 if data_type == 'train' else 10000

    with open(label_path, 'r') as f:
        parser = ijson.items(f, 'item')
        for item in tqdm(parser, total=total_items):
            image_name = item['name']
            
            valid_labels = []
            # 遍历一张图片上的所有标签
            for label in item['labels']:
                # 只保留那些是我们关心的类别，并且有box2d信息的标签
                if label.get('category') in BDD_CATEGORIES and label.get('box2d'):
                    valid_labels.append(label)
            
            # 只有当这张图片至少包含一个有效标签时，我们才把它加入最终的字典
            if valid_labels:
                image_to_labels_filtered[image_name] = valid_labels

    print(f"筛选完成！正在将 {len(image_to_labels_filtered)} 条有效数据保存到 '{pickle_path}' ...")
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(image_to_labels_filtered, f)
        
    print(f"成功保存！--- {data_type} 数据处理完毕 ---\n")

if __name__ == '__main__':
    process_and_save('train')
    process_and_save('val')
    print("所有数据预处理和筛选完成！")