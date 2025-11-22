# -*- coding: utf-8 -*-

import os
from lxml import etree

yolo_label_path = '/home/qk/data/new_test/labels'
xml_output_path = '/home/qk/data/new_test/Annotations'
image_path = '/home/qk/data/new_test/images'

# 创建xml标签存放路径
if not os.path.exists(xml_output_path):
    os.mkdir(xml_output_path)

# 手动指定数字到类别的映射字典
# class_dict = {'0': 'short sleeve top',
#               '1': 'long sleeve top',
#               '2': 'short sleeve outwear',
#               '3': 'long sleeve outwear',
#               '4': 'vest',
#               '5': 'sling',
#               '6': 'shorts',
#               '7': 'trousers',
#               '8': 'skirt',
#               '9': 'short sleeve dress',
#               '10': 'long sleeve dress',
#               '11': 'vest dress',
#               '12': 'sling dress',
#               '13': 'person',
#               '14': 'helmet'}

class_dict = {'0': 'hanging insulator',
              '1': 'post insulator',
              '2': 'column insulator',
              '3': 'wing insulator',
}

def convert_yolo2xml(file_name, yolo_label_path, xml_output_path, class_dict, image_path):
    # 读取图片尺寸
    image_file = file_name.replace('.txt', '.jpg')
    image = os.path.join(image_path, image_file)
    if not os.path.exists(image):
        image_file = file_name.replace('.txt', '.png')
        image = os.path.join(image_path, image_file)
        if not os.path.exists(image):
            print(f"Image {image_file} not found.")
            return
    
    from PIL import Image
    img = Image.open(image)
    width, height = img.size

    # 创建xml根节点
    annotation = etree.Element('annotation')

    # 文件名和路径
    etree.SubElement(annotation, 'filename').text = image_file
    etree.SubElement(annotation, 'path').text = image

    # 图片大小
    size = etree.SubElement(annotation, 'size')
    etree.SubElement(size, 'width').text = str(width)
    etree.SubElement(size, 'height').text = str(height)
    etree.SubElement(size, 'depth').text = str(3)  # Assuming RGB images

    # 读取YOLO格式标签文件
    with open(os.path.join(yolo_label_path, file_name), 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            label_class = class_dict.get(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height

            xmin = int(x_center - w / 2)
            ymin = int(y_center - h / 2)
            xmax = int(x_center + w / 2)
            ymax = int(y_center + h / 2)

            # 创建对象节点
            obj = etree.SubElement(annotation, 'object')
            etree.SubElement(obj, 'name').text = label_class
            etree.SubElement(obj, 'pose').text = 'Unspecified'
            etree.SubElement(obj, 'truncated').text = '0'
            etree.SubElement(obj, 'difficult').text = '0'

            # 边界框
            bndbox = etree.SubElement(obj, 'bndbox')
            etree.SubElement(bndbox, 'xmin').text = str(xmin)
            etree.SubElement(bndbox, 'ymin').text = str(ymin)
            etree.SubElement(bndbox, 'xmax').text = str(xmax)
            etree.SubElement(bndbox, 'ymax').text = str(ymax)

    # 将xml写入文件
    tree = etree.ElementTree(annotation)
    xml_file = file_name.replace('.txt', '.xml')
    tree.write(os.path.join(xml_output_path, xml_file), pretty_print=True, xml_declaration=True, encoding='utf-8')

# 获取文件名称列表
files = os.listdir(yolo_label_path)

# 对每个文件调用convert_yolo2xml函数，实现YOLO到XML的转换
for file in files:
    convert_yolo2xml(file, yolo_label_path, xml_output_path, class_dict, image_path)
