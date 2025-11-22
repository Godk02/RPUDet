# -*- coding: utf-8 -*-

import os
from lxml import etree

source_path = '/home/qk/data/VOC/test/testxml'
label_path = '/home/qk/data/VOC/test/labels'

# 创建txt标签存放路径
if not os.path.exists(label_path):
    os.mkdir(label_path)

# 手动指定类别到数字的映射字典
class_dict = {'aeroplane':'0',
            'bicycle':'1',
            'bird':'2',
            'boat':'3',
            'bottle':'4',
            'bus':'5',
            'car':'6',
            'cat':'7',
            'chair':'8',
            'cow':'9',
            'diningtable':'10',
            'dog':'11',
            'horse':'12',
            'motorbike':'13',
            'person':'14',
            'pottedplant':'15',
            'sheep':'16',
            'sofa':'17',
            'train':'18',
            'tvmonitor':'19',
}

def convert_xml2txt(file_name, source_path, label_path, class_dict, norm=False):
    # 创建txt文件，并打开、写入
    new_name = file_name.split('.')[0] + '.txt'
    f = open(os.path.join(label_path, new_name), 'w')
    with open(os.path.join(source_path, file_name), 'rb') as fb:
        # 开始解析xml文件，获取图像尺寸
        xml = etree.HTML(fb.read())
        width = int(xml.xpath('//size/width/text()')[0])
        height = int(xml.xpath('//size/height/text()')[0])
        # 获取对象标签
        labels = xml.xpath('//object') # 单张图片中的目标数量 len(labels)
        for label in labels:
            name = label.xpath('./name/text()')[0]
            label_class = class_dict.get(name)
            if label_class is not None:  # 确保类别在指定的类别字典中
                xmin = int(label.xpath('./bndbox/xmin/text()')[0])
                xmax = int(label.xpath('./bndbox/xmax/text()')[0])
                ymin = int(label.xpath('./bndbox/ymin/text()')[0])
                ymax = int(label.xpath('./bndbox/ymax/text()')[0])
                # xyxy-->xywh, 且归一化
                if norm:
                    dw = 1 / width
                    dh = 1 / height
                    x_center = (xmin + xmax) / 2
                    y_center = (ymax + ymin) / 2
                    w = (xmax - xmin)
                    h = (ymax - ymin)
                    x, y, w, h = x_center * dw, y_center * dh, w * dw, h * dh
                    f.write(f"{label_class} {x} {y} {w} {h}\n")
                else:
                    f.write(f"{label_class} {xmin} {ymin} {xmax} {ymax}\n")
    # 关闭文件
    f.close()

# 获取文件名称列表
files = os.listdir(source_path)

# 对每个文件调用convert_xml2txt函数，实现XML到TXT的转换
for file in files:
    convert_xml2txt(file, source_path, label_path, class_dict, norm=True)

