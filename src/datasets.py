import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import random

# dataset
from torch.utils.data import Dataset#, DataLoader
#from utils.collate_fn import detr_collate_fn

# from glob import glob
import os
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# root_dir = "/workspace/traffic_light/data/detection/train/"

import os
from transformers import AutoImageProcessor

random.seed(42)


class TrafficLightDataset(Dataset):
    """
    self.img_keys : The index and image_id are different, 
        so a mapping function is needed to map the index to the image_id.
    """
    def __init__(self, root_dir, transform=None):
        self.img_paths = os.path.join(root_dir, 'images') # img_paths : path
        self.coco = COCO(os.path.join(root_dir, 'train.json')) # json
        self.img_keys = sorted(self.coco.imgs.keys())
        self.transform = transform
        self.checkpoint = "facebook/detr-resnet-50"
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)

    def __call__(self, transform):
        self.transform = transform
        return self
    
    def __len__(self):
        return len(self.coco.imgs.keys())

    def _load_image(self,id: int) -> Image.Image:
        img_path = os.path.join(self.img_paths, self.coco.loadImgs(id)[0]['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        return image
        
    def _load_bbox(self,id):
        bboxes, labels, area = [], [], []
        
        for anno_id in self.coco.getAnnIds(id):
            anns = self.coco.loadAnns(anno_id)[0]
            labels.append(anns['category_id'])    
            bboxes.append(anns['bbox'])
            area.append(anns['area'])
        return bboxes, labels, area

    def formatted_anns(self, id, labels, area, bboxes, orig_w, orig_h):
        annotations = []
        for i in range(0, len(labels)):
            bbox = [round(value, 3) for value in bboxes[i]]
            new_ann = {
                "image_id": id,
                "category_id": labels[i],
                "isCrowd": 0,
                "area": area,
                "bbox": bbox,
                "orig_size": [orig_w, orig_h]
            }
            annotations.append(new_ann)
    
        return annotations

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        """
        데이터 양식
        annotations=["image_id": id, 
                     "annotations":[{'image_id': image_id, 
                                     'category_id':label,
                                     'isCrowd':0, 
                                     'area':area, 
                                     'bbox':[x,y,w,h]},
                                    {'image_id': image_id, 
                                     'category_id':label,
                                     'isCrowd':0, 
                                     'area':area, 
                                     'bbox':[x,y,w,h]}
                                   ]
                    ]

        target = {'pixel_values': tensor([[[[-2.1179, -2.1179, -2.1179,  ..., -2.0837, -1.8953, -2.1008],...]]]),
                  'pixel_mask' : tensor([[[1, 1, 1,  ..., 1, 1, 1], ...]]),
                  'labels' : [{'size': tensor([1089,  800]), 
                               'image_id': tensor([0]), 
                               'class_labels': tensor([4, 4]), 
                               'boxes': tensor([[0.4049, 0.2915, 0.1662, 0.1709],
                                                [0.4907, 0.5620, 0.1915, 0.1436]]), 
                               'area': tensor([24748.4570, 23948.5859]), 
                               'iscrowd': tensor([0, 0]), 
                               'orig_size': tensor([1024,  752])}]
                 }

        Returns:
            image: torch.Tensor[C, H, W]
            target: dict[str, torch.Tensor]
                "boxes": torch.float32 -> [x, y, w, h]
                "labels": torch.int64,
                "area": torch.float32
                "iscrowd": torch.int64
                "orig_size": torch.int64 -> [W, H]

        

        """
        index = self.img_keys[index]
        orig_width = self.coco.loadImgs(index)[0]['width']
        orig_height = self.coco.loadImgs(index)[0]['height']
        image = self._load_image(index) # type of image is np.array
        bboxes, labels, area = self._load_bbox(index) # type of these are python

        if self.transform != None:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image, bboxes = transformed['image'], transformed['bboxes']
        
        annotations = {"image_id": index, "annotations": self.formatted_anns(index, labels, area, bboxes, orig_width, orig_height)}
        encoding = self.image_processor(images=image, annotations=annotations, return_tensors="pt")

        
        image = encoding["pixel_values"].squeeze()
        mask = encoding["pixel_mask"].squeeze()
        labels = encoding["labels"][0] # remove batch dimension

        target = {'pixel_mask': mask, 'labels': labels}

        # target = {}
        # target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
        # target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        # target['area'] = torch.as_tensor(area, dtype=torch.float32)
        # target['iscrowd'] = torch.zeros(len(bboxes), dtype=torch.int64)
        # # TODO: 
        # target["image_id"] = torch.tensor([index], dtype=torch.int64)
        # target["orig_size"] = torch.tensor([img_size_width, img_size_height], dtype=torch.int64)
            
        return image, target
    

# data_loader = DataLoader(TrafficLightDataset(TRAIN_AUGMENTS_FOR_HUGGINGFACE), batch_size=2, shuffle=True, collate_fn=collate_fn)#lambda x: tuple(zip(*x)))
# data_loader = DataLoader(TrafficLightDataset("", ), batch_size=4, shuffle=True, collate_fn=detr_collate_fn)#lambda x: tuple(zip(*x)))

# print(next(iter(data_loader)))