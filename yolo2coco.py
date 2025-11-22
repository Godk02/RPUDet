# import os
# import json
# from PIL import Image

# # 文件路径设置
# coco_format_save_path = '/home/qk/data/vedai/VEDAI/'  # COCO格式标签保存路径
# yolo_format_annotation_path = '/home/qk/data/vedai/VEDAI/labels'  # YOLO格式标签路径
# img_pathDir = '/home/qk/data/vedai/VEDAI/images'  # 图片路径

# # 类别设置
# #class_names = ["hanging insulator", "post insulator", "column insulator", "wing insulator"]  # 修改为自己的类别
# # class_names = ['pedestrian',
# #     'people',
# #     'bicycle',
# #     'car',
# #     'van',
# #     'truck',
# #     'tricycle',
# #     'awning-tricycle',
# #     'bus',
# #     'motor']  # 修改为自己的类别
# class_names = ['car', 'pickup', 'camping','truck', 'other', 'tractor', 'boat', 'van' ]  # 修改为自己的类别

# categories = [{"id": idx, "name": name, "supercategory": ""} for idx, name in enumerate(class_names)]

# # 初始化COCO格式JSON
# coco_data = {
#     "info": {
#         "description": "Dataset converted from YOLO to COCO",
#         "version": "1.0",
#         "year": 2024,
#         "date_created": "2024-12-26"
#     },
#     "licenses": [],
#     "images": [],
#     "annotations": [],
#     "categories": categories
# }

# # 排序工具：根据文件名中的数字排序
# def sort_list_by_number(file_list):
#     def extract_number(s):
#         try:
#             return int(''.join(filter(str.isdigit, s)))
#         except ValueError:
#             return float('inf')  # 返回最大值以跳过无效文件
#     return sorted(file_list, key=extract_number)

# # 获取并排序图片文件列表
# image_files = [f for f in os.listdir(img_pathDir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
# image_files = sort_list_by_number(image_files)

# # 转换数据
# annotation_id = 1
# for image_id, image_file in enumerate(image_files):
#     # 获取图片路径和尺寸
#     image_path = os.path.join(img_pathDir, image_file)
#     with Image.open(image_path) as img:
#         width, height = img.size

#     # 添加图像信息
#     coco_data["images"].append({
#         "id": image_id,
#         "file_name": image_file,
#         "width": width,
#         "height": height
#     })

#     # 对应的YOLO格式标签文件
#     yolo_file = os.path.join(yolo_format_annotation_path, image_file.replace('.png', '.txt'))
#     if not os.path.exists(yolo_file):
#         continue

#     with open(yolo_file, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             # 解析YOLO标签
#             parts = line.split()
#             if len(parts) != 5:
#                 continue

#             class_id, x, y, w, h = map(float, parts)
#             class_id = int(class_id)

#             # 检查class_id合法性
#             if class_id < 0 or class_id >= len(class_names):
#                 print(f"Warning: Invalid class_id {class_id} in file {yolo_file}")
#                 continue

#             # 坐标转换
#             x_min = (x - w / 2) * width
#             y_min = (y - h / 2) * height
#             bbox_width = w * width
#             bbox_height = h * height

#             # 添加标注信息
#             coco_data["annotations"].append({
#                 "id": annotation_id,
#                 "image_id": image_id,
#                 "category_id": class_id,
#                 "bbox": [x_min, y_min, bbox_width, bbox_height],
#                 "area": bbox_width * bbox_height,
#                 "iscrowd": 0,
#                 "segmentation": []
#             })
#             annotation_id += 1

# # 保存COCO格式JSON
# os.makedirs(coco_format_save_path, exist_ok=True)
# output_file = os.path.join(coco_format_save_path, "annotations.json")
# with open(output_file, 'w') as f:
#     json.dump(coco_data, f, indent=4)

# print(f"COCO annotations saved to {output_file}")
import os
import json
from PIL import Image

# 文件路径设置
coco_format_save_path = '/home/qk/data/vedai1/val8/'  # COCO格式标签保存路径
yolo_format_annotation_path = '/home/qk/data/vedai1/val8/labels'  # YOLO格式标签路径
img_pathDir = '/home/qk/data/vedai/VEDAI/images'  # 图片路径

# 类别设置
class_names = ['car', 'pickup', 'camping', 'truck', 'other', 'tractor', 'boat', 'van']  # 修改为自己的类别

categories = [{"id": idx, "name": name, "supercategory": ""} for idx, name in enumerate(class_names)]

# 初始化COCO格式JSON
coco_data = {
    "info": {
        "description": "Dataset converted from YOLO to COCO",
        "version": "1.0",
        "year": 2024,
        "date_created": "2024-12-26"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": categories
}

# 排序工具：根据文件名中的数字排序
def sort_list_by_number(file_list):
    def extract_number(s):
        try:
            return int(''.join(filter(str.isdigit, s)))
        except ValueError:
            return float('inf')  # 返回最大值以跳过无效文件
    return sorted(file_list, key=extract_number)

# 获取并排序图片文件列表
image_files = [f for f in os.listdir(img_pathDir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
image_files = sort_list_by_number(image_files)

# 转换数据
annotation_id = 1
for image_file in image_files:
    # 获取图片路径和尺寸
    image_path = os.path.join(img_pathDir, image_file)
    with Image.open(image_path) as img:
        width, height = img.size

    # 从图片文件名中提取数字作为 image_id
    image_id = int(''.join(filter(str.isdigit, image_file.split('.')[0])))  # 提取文件名中的数字部分作为 image_id

    # 添加图像信息
    coco_data["images"].append({
        "id": image_id,
        "file_name": image_file,
        "width": width,
        "height": height
    })

    # 对应的YOLO格式标签文件
    yolo_file = os.path.join(yolo_format_annotation_path, image_file.replace('.png', '.txt'))
    if not os.path.exists(yolo_file):
        continue

    with open(yolo_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 解析YOLO标签
            parts = line.split()
            if len(parts) != 5:
                continue

            class_id, x, y, w, h = map(float, parts)
            class_id = int(class_id)

            # 检查class_id合法性
            if class_id < 0 or class_id >= len(class_names):
                print(f"Warning: Invalid class_id {class_id} in file {yolo_file}")
                continue

            # 坐标转换
            x_min = (x - w / 2) * width
            y_min = (y - h / 2) * height
            bbox_width = w * width
            bbox_height = h * height

            # 添加标注信息
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1

# 保存COCO格式JSON
os.makedirs(coco_format_save_path, exist_ok=True)
output_file = os.path.join(coco_format_save_path, "annotations.json")
with open(output_file, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations saved to {output_file}")
