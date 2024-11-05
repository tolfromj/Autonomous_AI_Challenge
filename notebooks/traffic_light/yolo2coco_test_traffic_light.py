import os
import json
from tqdm import tqdm
from PIL import Image

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
        "contributor": "369", 
        "url": "", 
        "date_created": "2024-10-22 15:00:00",
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

def make_test_coco(output_dir):
    img_path = os.path.join(output_dir, 'images')
    images = sorted(os.listdir(img_path))


    for image in tqdm(images):
        file_name = os.path.splitext(image)[0]
        with Image.open(os.path.join(img_path, image)) as img:
            original_w, original_h = img.size
        image_dict = {
            "id": int(file_name),
            "width": int(original_w),
            "height": int(original_h),
            "file_name": image, # train/file_name.jpg
            "license": 0,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None,
        }
        
        coco_dataset["images"].append(image_dict)

    # make coco_datset.json
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(coco_dataset, f)

if __name__=='__main__':
    output_dir='' # path/to/test folder e.g) /workspace/datasets/data_traffic/test
    make_test_coco(output_dir)
