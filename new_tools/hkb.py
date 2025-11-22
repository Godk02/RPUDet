import cv2
import numpy as np
import os

# 定义一个函数用于绘制矩形框
def draw_boxes(image, label_file, color_map=None):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    h, w, _ = image.shape
    for line in lines:
        # YOLO格式：class_id x_center y_center width height
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        
        # 只标注类别ID为2（bicycle）和9（motor）的物体
        if class_id not in [2, 9]:  # 如果不是bicycle（2）或者motor（9），跳过
            continue
        
        # 将YOLO坐标转换为像素坐标
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        
        # 计算框的左上角和右下角
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # 根据类别ID获取颜色
        color = color_map.get(int(class_id), (0, 255, 0))  # 默认为绿色

        # 画框
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)  # 1表示框较细

    return image

# 主函数
def process_images(image_folder, label_folder, output_folder, color_map=None):
    # 如果输出文件夹不存在，创建文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历图像文件夹
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # 读取图片
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            # 获取对应的标签文件路径
            label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(label_folder, label_filename)

            # 如果标签文件存在
            if os.path.exists(label_path):
                # 在图片上绘制框
                output_image = draw_boxes(image, label_path, color_map)
                # 保存处理后的图像
                output_image_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_image_path, output_image)

# 示例使用
image_folder = '/home/qk/data/VisDrone2019-DET-val/images/'  # 图像文件夹路径
label_folder = '/home/qk/data/VisDrone2019-DET-val/labels'  # 标签文件夹路径
output_folder = '/home/qk/data/VisDrone2019-DET-val/outputb'  # 输出文件夹路径

# 自定义类别颜色映射 (2: bicycle, 9: motor)
color_map = {
    2: (0, 255, 0),    # 绿色 (bicycle)
    9: (255, 105, 180),# 熏衣草粉 (motor)
}

process_images(image_folder, label_folder, output_folder, color_map)
