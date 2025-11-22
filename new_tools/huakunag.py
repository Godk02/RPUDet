# import cv2
# import numpy as np
# import os

# # 定义一个函数用于绘制矩形框
# def draw_boxes(image, label_file, color_map=None):
#     with open(label_file, 'r') as f:
#         lines = f.readlines()

#     h, w, _ = image.shape
#     for line in lines:
#         # YOLO格式：class_id x_center y_center width height
#         class_id, x_center, y_center, width, height = map(float, line.strip().split())
        
#         # 将YOLO坐标转换为像素坐标
#         x_center *= w
#         y_center *= h
#         width *= w
#         height *= h
        
#         # 计算框的左上角和右下角
#         x1 = int(x_center - width / 2)
#         y1 = int(y_center - height / 2)
#         x2 = int(x_center + width / 2)
#         y2 = int(y_center + height / 2)
        
#         # 如果没有传入颜色映射，使用默认颜色映射
#         if color_map is None:
#             color = tuple(np.random.randint(0, 256, 3).tolist())  # 随机颜色
#         else:
#             color = color_map.get(int(class_id), (0, 255, 0))  # 默认绿色

#         # 画框
#         image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)  # 1表示框较细

#     return image

# # 主函数
# def process_images(image_folder, label_folder, output_folder, color_map=None):
#     # 如果输出文件夹不存在，创建文件夹
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 遍历图像文件夹
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.jpg', '.png', '.jpeg')):
#             # 读取图片
#             image_path = os.path.join(image_folder, filename)
#             image = cv2.imread(image_path)

#             # 获取对应的标签文件路径
#             label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
#             label_path = os.path.join(label_folder, label_filename)

#             # 如果标签文件存在
#             if os.path.exists(label_path):
#                 # 在图片上绘制框
#                 output_image = draw_boxes(image, label_path, color_map)
#                 # 保存处理后的图像
#                 output_image_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_image_path, output_image)

# # 示例使用
# image_folder = '/home/qk/data/vedai/VEDAI/images/'  # 图像文件夹路径
# label_folder = '/home/qk/data/vedai/VEDAI/labels'  # 标签文件夹路径
# output_folder = '/home/qk/data/vedai/VEDAI/output'  # 输出文件夹路径

# # 自定义类别颜色映射 (例如：0 -> 红色, 1 -> 蓝色)
# color_map = {0: (0, 0, 255), 1: (255, 0, 0)}  # 这里可以根据需要修改颜色

# process_images(image_folder, label_folder, output_folder, color_map)
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
output_folder = '/home/qk/data/VisDrone2019-DET-val/output'  # 输出文件夹路径
# 自定义类别颜色映射 (0-9十个类别)，颜色选择更为明显的区别
color_map = {
    0: (0, 0, 255),    # 红色 (pedestrian)
    1: (255, 0, 0),    # 蓝色 (people)
    2: (0, 255, 0),    # 绿色 (bicycle)
    3: (0, 255, 255),  # 黄色 (car)
    4: (255, 165, 0),  # 橙色 (van)
    5: (128, 0, 128),  # 紫色 (truck)
    6: (255, 192, 203),# 粉色 (tricycle)
    7: (0, 128, 128),  # 暗青 (awning-tricycle)
    8: (128, 128, 0),  # 橄榄绿 (bus)
    9: (255, 105, 180),# 熏衣草粉 (motor)
}

process_images(image_folder, label_folder, output_folder, color_map)
