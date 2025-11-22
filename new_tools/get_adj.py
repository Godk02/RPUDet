import glob
import os
import csv
from pathlib import Path
import numpy as np
import pickle

# 类别名称
classes = [
    # "Suspension insulator",
    # "Pin insulator",
    # "Pillar insulator",
    # "Butterfly insulator",
    # "Isolation knife",
    # "Transformer",
    # "Pole"

    # 'hanging insulator',
    # 'post insulator',
    # 'column insulator',
    # 'wing insulator'

    # 'pedestrian',
    # 'people',
    # 'bicycle',
    # 'car',
    # 'van',
    # 'truck',
    # 'tricycle',
    # 'awning-tricycle',
    # 'bus',
    # 'motor',

    # 'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 
    # 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool' 

    'vehicle','vehicle'
]

# 定义csv表头
keys = ['img_name'] + classes

# 图像文件格式
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

# 标签路径转换
def img2label_paths(img_paths):
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

# 加载图像和标签
def LoadImagesAndLabels(path):
    f = []
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # 如果是目录
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():  # 如果是文件
            with open(p, 'r') as t:
                t = t.read().strip().splitlines()
                for x in t:
                    f += [x]
        else:
            raise Exception(f'{p} does not exist')
    
    img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
    label_files = img2label_paths(img_files)
    return img_files, label_files

# 统计类别信息并写入CSV
def statistics(img_files, label_files, classes, save_path, flag=0):
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if flag == 0:
            writer.writeheader()  # 如果是第一次写入表头
            flag += 1
        
        # 初始化字典
        for i in range(len(label_files)):
            dic = {'img_name': img_files[i]}
            dic.update({cls: 0 for cls in classes})  # 将每个类别初始值设为0

            print(f"Processing: {img_files[i]}")  # 输出正在处理的图像文件

            # 读取标签文件
            try:
                with open(label_files[i]) as f1:
                    while True:
                        line = f1.readline()
                        if not line:
                            break
                        # 获取类别索引
                        class_index = int(line.split()[0])  # YOLO格式中的类别索引是每行的第一个元素
                        if 0 <= class_index < len(classes):  # 确保类别索引在范围内
                            dic[classes[class_index]] = 1  # 标记该类别在图像中出现
            except FileNotFoundError:
                print(f"Label file not found: {label_files[i]}")  # 如果标签文件未找到，进行异常处理

            writer.writerow(dic)

# 创建类别邻接矩阵并保存为pickle文件
def make_adj_file(train_csv, classes, pick_save_path):
    # 检查文件是否存在并且非空
    if not os.path.exists(train_csv) or os.path.getsize(train_csv) == 0:
        raise ValueError(f"The file {train_csv} is empty or does not exist.")

    try:
        tmp = np.loadtxt(train_csv, dtype=str, delimiter=',')
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # 如果文件成功加载，检查其形状
    if tmp.ndim == 1:  # 如果文件被错误地加载为1D
        print("The loaded data is 1D, but we expected a 2D array.")
        return

    times = tmp[1:, 1:]  # 跳过表头行并处理数据
    adj_matrix = np.zeros(shape=(len(classes), len(classes)))
    nums_matrix = np.zeros(shape=(len(classes)))

    for index in range(len(times)):
        data = times[index]
        for i in range(len(classes)):
            if int(data[i]) >= 1:  # 如果该类在该图像中出现
                nums_matrix[i] += 1
                for j in range(len(classes)):
                    if j != i and int(data[j]) >= 1:
                        adj_matrix[i][j] += 1  # 记录类别i和j的共同出现

    adj = {'adj': adj_matrix, 'nums': nums_matrix}
    print(adj)  # 打印邻接矩阵和类别计数
    pickle.dump(adj, open(pick_save_path, 'wb'), pickle.HIGHEST_PROTOCOL)  # 保存结果为pickle文件


if __name__ == '__main__':
    path = '/home/qk/data/usod/vehicle/'  # 训练集路径
    save_path = '/home/qk/data/usod/vehicle/statics.csv'  # 统计结果保存路径
    pick_save_path = 'usod_adj_matrix.pkl'  # 邻接矩阵保存路径

    # 加载图像文件和标签文件
    img_files, label_files = LoadImagesAndLabels(path)

    # 统计类别信息并写入CSV
    statistics(img_files=img_files, label_files=label_files, classes=classes, save_path=save_path)

    # 创建邻接矩阵并保存
    make_adj_file(save_path, classes, pick_save_path)
