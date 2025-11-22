import cv2
import numpy as np
import os

def load_yolo_labels(label_file, img_width, img_height):
    boxes = []
    class_ids = []
    with open(label_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            boxes.append((x1, y1, x2, y2))
            class_ids.append(int(class_id))
    return boxes, class_ids

def draw_boxes(image, boxes, class_ids):
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制矩形框
        label = f'Class {class_id}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # 添加标签
    return image

def main(img_path, label_path):
    # 读取图像
    image = cv2.imread(img_path)
    img_height, img_width = image.shape[:2]

    # 加载标签
    boxes, class_ids = load_yolo_labels(label_path, img_width, img_height)

    # 绘制边界框
    output_image = draw_boxes(image.copy(), boxes, class_ids)

    # 保存结果
    output_path = os.path.splitext(img_path)[0] + '_output.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"Output image saved to: {output_path}")

# 使用示例
img_path = "new_mix/enhanced_image.jpg"  # 输入图像路径
label_path = "new_mix/enhanced_labels.txt"  # YOLO 标签路径

main(img_path, label_path)



