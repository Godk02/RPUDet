# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0  
image_id = 20180000000
annotation_id = 0

def addCatItem(name, category_id):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id = category_id
    category_item['id'] = category_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_id
    return category_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    seg.append(bbox[0])
    seg.append(bbox[1])
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids

def parseXmlFiles(xml_path, json_save_path, category_map):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue
        
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = f.split(".")[0] + ".jpg"

            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        if object_name in category_map:
                            current_category_id = addCatItem(object_name, category_map[object_name])
                        else:
                            raise Exception(f'Category ID for {object_name} not found in category_map.')
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                   
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    bbox.append(bndbox['xmin'])
                    bbox.append(bndbox['ymin'])
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)
    json.dump(coco, open(json_save_path, 'w'))

if __name__ == '__main__':
    category_map = {
        #"xuanshi-jyz": 1,
        #"zhenshi-jyz": 2,
        #"zhushi-jyz": 3,
        #"dieshi-jyz": 4,
        #"zssb-dlqwzgld": 5,
        #"zstb-s11byq": 6,
        #"dxg": 7,
        # Add more category names and their corresponding IDs as needed
        #"baliser_ok":1,
        #"baliser_aok":2,
        #"baliser_nok":3,
        #"insulator_ok":4,
        #"insulator_nok":5,
        #"bird_nest":6,
        #"stockbridge_ok":7,
        #"stockbridge_nok":8,
        #"spacer_ok":9,
        #"spacer_nok":10,
        #"insulator_unk":11,
        
        # "Suspension insulator": 1,
        # "Pin insulator": 2,
        # "Pillar insulator": 3,
        # "Butterfly insulator": 4,
        # "Isolation knife": 5,
        # "Transformer": 6,
        # "Pole": 7,
       "hanging insulator": 0,
        "post insulator": 1,
        "column insulator": 2,
        "wing insulator": 3,
        
    }
    ann_path = '/home/qk/data/new_test/Annotations'
    json_save_path = "/home/qk/data/new_test/annotations.json"
    parseXmlFiles(ann_path, json_save_path, category_map)
