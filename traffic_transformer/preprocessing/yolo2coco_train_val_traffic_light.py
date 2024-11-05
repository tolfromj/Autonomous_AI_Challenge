import os
import json

import pandas as pd
from tqdm import tqdm
import glob
from PIL import Image
from pycocotools.coco import COCO


def modify_annos(root_path, output_file_name, anno_id, box):
    with open(os.path.join(root_path, f'{output_file_name}.json'), 'r') as f:
        coco_data = json.load(f)
        for annotation in coco_data['annotations']:
            if annotation['id'] == anno_id:
                annotation['bbox'] = box

    with open(os.path.join(root_path, f'{output_file_name}.json'), 'w') as f:
        json.dump(coco_data, f, indent=4)

def data_cleansing(root_path):
    output_file_name = os.path.basename(root_path)
    coco = COCO(os.path.join(root_path, f'{output_file_name}.json'))

    cnt = 0
    for _id in tqdm(sorted(coco.getImgIds())):
        # _id = 276
        # print(coco.loadImgs(_id))
        w = coco.loadImgs(_id)[0]['width']
        h = coco.loadImgs(_id)[0]['height']
        # print(coco.getAnnIds(_id))
        for i in coco.getAnnIds(_id):
            need_modify = False
            bbox = coco.loadAnns(i)[0]['bbox']
            x_max = bbox[0]+bbox[2]
            x_min = bbox[0]
            y_max = bbox[1]+bbox[3]
            y_min = bbox[1]
            if x_max > w:
                # print()
                # print('x_max', coco.loadAnns(i))
                bbox[2] = float(w-bbox[0])
                need_modify = True
                cnt +=1
            if x_min < 0:
                # print(f'(w:{w},h:{h}')
                # print('x_min',coco.loadAnns(i))
                bbox[0] = float(0)
                need_modify = True
                cnt +=1
            if y_max > h:
                # print()
                # print('y_max',coco.loadAnns(i))
                bbox[3] = float(h-bbox[1])
                need_modify = True
                cnt +=1
            if y_min < 0:
                # print(f'(w:{w},h:{h}')
                # print('y_min', coco.loadAnns(i))
                bbox[1] = float(0)
                need_modify = True
                cnt +=1
            if need_modify:
                modify_annos(root_path, output_file_name, i, bbox)
    print(f'Number of cleansed {output_file_name} data files: {cnt}')

def make_dataframe(img_path:str, labels_path: str)-> pd.DataFrame:
    df = pd.DataFrame({
        'file_name': [],
        'class_id': [],
        'norm_center_x': [],
        'norm_center_y': [],
        'norm_w': [],
        'norm_h': [],
        'original_w': [],
        'original_h': [],
    })

    # file_names = sorted(os.listdir(labels_path))
    files = sorted(glob.glob(f'{labels_path}/*.txt'))
    # file names : [... ,'00000036.txt', '00000037.txt', ...]
    # file_name : '00000036.txt'
    # name : '00000036'
    cnt = 0
    for file in tqdm(files):
        name = os.path.splitext(os.path.basename(file))[0]
        # name = file_name.split('.txt')[0]
        with open(file, 'r') as file:
            with Image.open(os.path.join(img_path, f"{name}.jpg")) as img:
                original_w, original_h = img.size
            for line in file:
                class_id, norm_center_x, norm_center_y, norm_w, norm_h = line.strip().split()
                
                df.loc[cnt] = [
                    name,
                    int(class_id), 
                    float(norm_center_x), 
                    float(norm_center_y), 
                    float(norm_w), 
                    float(norm_h),
                    original_w,
                    original_h
                ]
                cnt += 1
    # print(df.head())
    # print(df.info())
    return df

def bbox_yolo2coco(ann):
    file_name, class_id, norm_center_x, norm_center_y, norm_w, norm_h, original_w, original_h = ann
    x = (norm_center_x - norm_w/2) * original_w
    y = (norm_center_y - norm_h/2) * original_h
    w = norm_w * original_w
    h = norm_h * original_h
    return [x, y, w, h]

def label_yolo2coco(root_path):
    categories = [
        {"id": 0, "name": "veh_go", "supercategory": "veh_go"},
        {"id": 1, "name": "veh_goLeft", "supercategory": "veh_goLeft"},
        {"id": 2, "name": "veh_noSign", "supercategory": "veh_noSign"},
        {"id": 3, "name": "veh_stop", "supercategory": "veh_stop"},
        {"id": 4, "name": "veh_stopLeft", "supercategory": "veh_stopLeft"},
        {"id": 5, "name": "veh_stopWarning", "supercategory": "veh_stopWarning"},
        {"id": 6, "name": "veh_warning", "supercategory": "veh_warning"},
        {"id": 7, "name": "ped_go", "supercategory": "ped_go"},
        {"id": 8, "name": "ped_noSign", "supercategory": "ped_noSign"},
        {"id": 9, "name": "ped_stop", "supercategory": "ped_stop"},
        {"id": 10, "name": "bus_go", "supercategory": "bus_go"},
        {"id": 11, "name": "bus_noSign", "supercategory": "bus_noSign"},
        {"id": 12, "name": "bus_stop", "supercategory": "bus_stop"},
        {"id": 13, "name": "bus_warning", "supercategory": "bus_warning"}
    ]

    coco_dataset = {
        "info": {
            "year": 2024, 
            "version": "1.0.0", 
            "description": "traffic light", 
            "contributor": "tlfromj", 
            "url": "", 
            "date_created": "2024-10-02 21:00:00",
        },
        "licenses": [{
            "id": 0, 
            "name": None, 
            "url": None,
        }],
        "categories": categories,
        "images": [],
        "annotations": []
    }
    
    img_path = f'{root_path}/images'
    labels_path = f'{root_path}/labels'
    anno_dicts = []
    df = make_dataframe(img_path, labels_path)
    print(df)
    print(df.head())
    print(df.info())
    for i in tqdm(range(len(df))):
        file_name, class_id, norm_center_x, norm_center_y, norm_w, norm_h, original_w, original_h = df.loc[i]
        
        image_dict = {
            "id": int(file_name),
            "width": int(original_w),
            "height": int(original_h),
            "file_name": f"{file_name}.jpg", # train/file_name.jpg
            "license": 0,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None,
        }
        
        coco_dataset["images"].append(image_dict)
        
        anno_dict = {
            "id": i,
            "image_id": int(file_name),
            "category_id": int(class_id),
            "area": float(norm_w * norm_h * original_w * original_h),
            "bbox": bbox_yolo2coco(df.loc[i]),
            "iscrowd": 0
        }
        anno_dicts.append(anno_dict)
    
    coco_dataset["annotations"].extend(anno_dicts)

    output_file_name = os.path.basename(root_path)
    output_path = os.path.join(root_path, f'{output_file_name}.json')
    
    # make coco datset.json
    with open(output_path, 'w') as f:
        json.dump(coco_dataset, f)
    print(f'created {output_path}')

if __name__=='__main__':
    root_dir='../../data/detection'  # path/to/dataset e.g) ./datasets/data_traffic/train
    train_root_path=os.path.join(root_dir, 'train0') 
    val_root_path=os.path.join(root_dir, 'val0')
    
    label_yolo2coco(train_root_path)
    label_yolo2coco(val_root_path)

    data_cleansing(train_root_path)
    data_cleansing(val_root_path)