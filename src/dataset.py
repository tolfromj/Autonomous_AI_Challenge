import torch
import numpy as np
from torch import nn
from tqdm import tqdm

# dataset
from torch.utils.data import Dataset, DataLoader
# from glob import glob
import os
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

root_folder = "/workspace/data/traffic_light/train/"

import os
from transformers import AutoImageProcessor

class CustomDataset(Dataset): # torchvision.datasets.CocoDetection 이것도 방법임!

    def __init__(self, root_folder, transform=None, augmentation=True, data_format='coco',): 
        self.img_paths = os.path.join(root_folder, 'images') # img_paths : path
        self.coco = COCO(os.path.join(root_folder, 'annotations.json')) # json
        self.transform = transform
        self.CHECKPOINT = "facebook/detr-resnet-50"  ################## 나중에 수정하기 #######
        self.image_processor = AutoImageProcessor.from_pretrained(self.CHECKPOINT)  ################## 나중에 수정하기 #######
        self.aug = augmentation
        self.data_format = data_format
        self.augmentation = A.Compose([
            # 스케일을 무작위로 변경하는 증강
            A.RandomScale(scale_limit=0.2, p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            # 이미지를 512x512 크기로 리사이징
            A.Resize(height=480, width=480, p=1.0),
            # PyTorch 모델에 입력하기 위한 변환
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        
    def __len__(self): # 샘플 수 반환
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
            if self.data_format == 'pascal_voc':
                bboxes.append(self._coco_to_pascal_bbox(anns['bbox']))
            else: bboxes.append(anns['bbox'])
            area.append(anns['area'])
        return bboxes, labels, area

    def _coco_to_pascal_bbox(self,anns):
        return [anns[0], anns[1], anns[0]+anns[2], anns[1]+anns[3]]
        
    def _augmetation(self, image, bboxes, labels):
        augmented = self.augmentation(image=image, bboxes=bboxes, labels=labels)
        return augmented['image'], augmented['bboxes']
    
    def formatted_anns(self, id, labels, area, bboxes):
        annotations = []
        for i in range(0, len(labels)):
            new_ann = {
                "image_id": id,
                "category_id": labels[i],
                "isCrowd": 0,
                "area": area,
                "bbox": bboxes[i],
            }
            annotations.append(new_ann)
    
        return annotations
    
    def __getitem__(self, id):
        img_size_width = self.coco.loadImgs(id)[0]['width']
        img_size_height = self.coco.loadImgs(id)[0]['height']
        image = self._load_image(id) # type of image is np.array
        bboxes, labels, area = self._load_bbox(id) # type of these are python
        
        if self.aug:
            image, bboxes = self._augmetation(image, bboxes, labels)
        
        annotations = {"image_id": id, "annotations": self.formatted_anns(id, labels, area, bboxes)}
        encoding = self.image_processor(images=image, annotations=annotations, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        
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
        """
        # target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
        # target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        # target['area'] = torch.as_tensor(area, dtype=torch.float32)
        # target['iscrowd'] = torch.zeros(len(bboxes), dtype=torch.int64)
            
        
        return pixel_values, target