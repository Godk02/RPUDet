import cv2
import numpy as np
import os

def load_yolo_labels(label_file):
    boxes = []
    class_ids = []
    with open(label_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_ids.append(int(class_id))  # 保留原始类ID
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            boxes.append((x1, y1, x2, y2))
    return boxes, class_ids

def crop_image(image, boxes):
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image, (x_min, y_min, x_max, y_max)

def update_labels(boxes, class_ids, offset):
    updated_boxes = []
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        x1 -= offset[0]
        y1 -= offset[1]
        x2 -= offset[0]
        y2 -= offset[1]
        updated_boxes.append((x1, y1, x2, y2, class_id))  # 保留原始类ID
    return updated_boxes

def save_cropped_image_and_labels(image, boxes, save_dir, image_name, label_name):
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存裁剪后的图像
    cropped_image_path = os.path.join(save_dir, image_name)
    cv2.imwrite(cropped_image_path, image)
    
    # 保存更新后的标签
    with open(os.path.join(save_dir, label_name), 'w') as f:
        for box in boxes:
            x_center = (box[0] + box[2]) / 2 / image.shape[1]
            y_center = (box[1] + box[3]) / 2 / image.shape[0]
            width = (box[2] - box[0]) / image.shape[1]
            height = (box[3] - box[1]) / image.shape[0]
            class_id = box[4]  # 获取原始类ID
            f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

# 使用示例
img_path = "/home/qk/data/new_test/images/10_2073.jpg"
label_path = "/home/qk/data/new_test/labels/10_2073.txt"
save_directory = 'caijian'  # 指定保存目录
image_name = 'cropped_image.jpg'
label_name = 'updated_labels.txt'

# 读取图像
image = cv2.imread(img_path)
img_height, img_width = image.shape[:2]

# 加载标签
boxes, class_ids = load_yolo_labels(label_path)

# 裁剪图像
cropped_image, (x_min, y_min, x_max, y_max) = crop_image(image, boxes)

# 更新标签
updated_labels = update_labels(boxes, class_ids, (x_min, y_min))

# 保存裁剪后的图像和更新后的标签
save_cropped_image_and_labels(cropped_image, updated_labels, save_directory, image_name, label_name)
