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



from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import AutoImageProcessor   ################## 나중에 수정하기 #######

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    https://github.com/victoresque/pytorch-template/blob/master/base/base_data_loader.py#L7
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split=0.0, num_workers=1, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        if collate_fn == 'detr':
            self.collate_fn = self._collate_fn
        else: self.collate_fn = collate_fn
            
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def _collate_fn(self, batch):
        CHECKPOINT = self.get_model_checkpoint()
        image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT) ################## 나중에 수정하기 #######
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        # batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch

    def get_model_checkpoint(self):  
        return "facebook/detr-resnet-50"  ################## 나중에 수정하기 #######
    





from torch import nn
from transformers import DetrForObjectDetection
def get_model(model_name):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_name == 'facebook/detr-resnet-50':
        return DetrModel()
        
    
class DetrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.class_labels_classifier = torch.nn.Linear(in_features=256, out_features=15, bias=True)
        self.name = 'facebook/detr-resnet-50'

    def forward(self, x):
        x = self.model(x)
        return x







import torch
from torch import nn
import torch.nn.functional as F


def get_loss(loss_name: str) -> nn.Module:
    """
    loss 얻는 method

    :param loss_name: loss name
    :type loss_name: str
    :return: loss
    :rtype: nn.Module
    """
    if loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'label_smooth':
        return LabelSmoothingLoss()
    elif loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'f1':
        return F1Loss()
    else:
        return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class F1Loss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
    








import random
from typing import Tuple

import numpy as np

import torch
from torch import Tensor, nn


_Optimizer = torch.optim.Optimizer


def seed_everything(seed: int) -> None:
    """
    시드 고정 method

    :param seed: 시드
    :type seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer: _Optimizer) -> float:
    """
    optimizer 통해 lr 얻는 method

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :return: learning_rate
    :rtype: float
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]







from typing import Dict

import torch
from torch import optim, nn

# from lion_pytorch import Lion


_Optimizer = torch.optim.Optimizer


def get_optim(optim_name: str, model: nn.Module, configs: Dict) -> _Optimizer:
    """
    optimizer 얻는 method

    :param loss_name: loss name
    :type loss_name: str
    :param model: model for obtain parameters
    :type model: nn.Module
    :param configs: config for lr
    :type configs: Dict
    :return: optimizer
    :rtype: torch.optim.Optimizer
    """
    optim_name = optim_name.lower()
    lr = configs['train']['lr']

    if optim_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    elif optim_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optim_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    # elif optim_name == 'lion':
    #     return Lion(model.parameters(), lr=lr)
    else:
        return optim.Adam(model.parameters(), lr=lr)








import argparse
import os
import multiprocessing
import random
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
# import wandb
from sklearn.metrics import f1_score, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# from datasets.mask_datasets import MaskSplitByProfileDataset
# from models.mask_model import get_model
# from utils.utils import mixup_aug, mixuploss, cutmix_aug, cutmixloss
# from utils.utils import get_lr, seed_everything
# from ops.losses import get_loss
# from ops.optim import get_optim

# import evaluate


import warnings
warnings.filterwarnings('ignore')

_Optimizer = torch.optim.Optimizer
_Scheduler = torch.optim.lr_scheduler._LRScheduler
scaler = GradScaler()


def train(
    configs: Dict, dataloader: DataLoader, device: str,
    model: nn.Module, loss_fn: nn.Module, optimizer: _Optimizer,
    scheduler: _Scheduler, epoch: int, # mix: str
) -> None:
    """
    데이터셋으로 훈련

    :param dataloader: PyTorch DataLoader
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 손실 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    :param scheduler: 훈련에 사용되는 스케줄러
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param epoch: 현재 훈련되는 epoch
    :type epoch: int
    :param mixup: mixup 사용 여부
    :type mixup: str
    """
    model.train()

    loss_value = 0
    train_loss = 0
    train_acc = 0
    accuracy = 0

    epochs = configs['train']['epoch']
    for batch, (images, targets) in enumerate(dataloader):
        images = images.float().to(device)
        targets = targets.long().to(device)
        # if mix == 'mixup' and (batch + 1) % 3 == 0:
        #     images, labels_a, labels_b, lambda_ = mixup_aug(images, targets)
        #     with autocast():
        #         outputs = model(images)
        #         loss = mixuploss(
        #             loss_fn, pred=outputs, l_a=labels_a,
        #             l_b=labels_b, lambda_=lambda_
        #         )
        # elif mix == 'cutmix' and (batch + 1) % 3 == 0:
        #     images, target_a, target_b, lambda_ = cutmix_aug(images, targets)
        #     with autocast():
        #         outputs = model(images)
        #         loss = cutmixloss(
        #             loss_fn, pred=outputs, l_a=target_a,
        #             l_b=target_b, lambda_=lambda_
        #         )
        # else:
        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value += loss.item()
        outputs = outputs.argmax(dim=-1)
        accuracy += (outputs == targets).sum().item()

        log_term = configs['train']['log_interval']
        if (batch+1) % log_term == 0:
            train_loss = loss_value / log_term
            train_acc = accuracy / configs['train']['batch_size'] / log_term
            current_lr = get_lr(optimizer)

            print(
                f"Epoch[{epoch}/{epochs}]({batch + 1}/{len(dataloader)}) "
                f"| lr {current_lr} \ntrain loss {train_loss:4.4}"
                f" | train acc {train_acc}"
            )

            loss_value = 0
            accuracy = 0

        # if not configs['fast_train_mode']:
        #     wandb.log({
        #             "train_loss": train_loss,
        #             "train_acc": train_acc,
        #             'train_rgb': wandb.Image(
        #                 images[0], caption=f'{targets[0]}'
        #             )
        #         }, step=epoch)

    if scheduler is not None:
        scheduler.step()


def validation(
    dataloader: DataLoader, save_dir: os.PathLike, device: str,
    model: nn.Module, loss_fn: nn.Module, epoch: int
) -> float:
    """
    데이터셋으로 검증

    :param dataloader: PyTorch DataLoader
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 손실 함수
    :type loss_fn: nn.Module
    :param epoch: 현재 훈련되는 epoch
    :type epoch: int
    :return: valid_loss
    :rtype: float
    """
    model.eval()

    valid_loss = []
    valid_acc = []
    val_labels = []
    val_preds = []
    example_images = []

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = images.float().to(device)
            targets = targets.long().to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            valid_loss.append(loss.item())

            outputs = outputs.argmax(dim=-1)
            val_acc_item = (outputs == targets).sum().item()
            valid_acc.append(val_acc_item)
            val_labels.extend(targets.cpu().numpy())
            val_preds.extend(outputs.cpu().numpy())

            if batch % configs['train']['log_interval'] == 0:
                idx = random.randint(0, outputs.size(0)-1)
                outputs = str(outputs[idx].cpu().numpy())
                targets = str(targets[idx].cpu().numpy())
                # if not configs['fast_train_mode']:
                #     example_images.append(wandb.Image(
                #         images[idx],
                #         caption="Pred: {} Truth: {}".format(outputs, targets)
                #     ))

    val_loss = np.sum(valid_loss) / len(dataloader)
    val_acc = np.sum(valid_acc) / len(dataloader.dataset)
    val_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average='macro')

    print(
        f"Epoch[{epoch}]({len(dataloader)})"
        f"valid loss {val_loss:4.4} | valid acc {val_acc:4.2%}"
        f"\nvalid f1 score {val_f1:.5}"
    )
    print(classification_report(y_true=val_labels, y_pred=val_preds))

    # if not configs['fast_train_mode']:
    #     wandb.log({
    #         "Image": example_images,
    #         "valid_loss": val_loss,
    #         "valid_acc": val_acc,
    #         "val_f1_score": val_f1
    #     }, step=epoch)

    torch.save(
        model.state_dict(),
        f'{save_dir}/{epoch}-{val_loss:4.4}-{val_acc:4.2}.pth'
        )
    print(
        f'Saved Model to {save_dir}/{epoch}-{val_loss:4.4}-{val_acc:4.2}.pth'
    )
    return val_loss


def run_pytorch(configs: Dict) -> None:
    """
    학습 파이토치 파이프라인

    :param configs: 학습에 사용할 config
    :type configs: dict
    """

    # if not configs['fast_train_mode']:
    #     wandb.init(
    #         project="level1-imageclassification-cv-07",
    #         entity='naver-ai-tech-cv07',
    #         config={
    #             'seed': configs['seed'],
    #             'model': configs['model'],
    #             'img_size': configs['data']['image_size'],
    #             'loss': configs['train']['loss'],
    #             'optim': configs['train']['optim'],
    #             'batch_size': configs['train']['batch_size'],
    #             'lr': configs['train']['lr'],
    #             'epoch': configs['train']['epoch'],
    #             'imagenet': configs['train']['imagenet'],
    #             'early_patience': configs['train']['early_patience'],
    #             'mix': configs['train']['mix'],
    #         }
    #     )

    dataset = CustomDataset(
        root_folder=configs['data']['train_dir'],
        # valid_rate=configs['data']['valid_rate'],
        # csv_path=configs['data']['csv_dir']
    )

    # width, height = map(int, configs['data']['image_size'].split(','))
    # if configs['train']['imagenet']:
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    # else:
    #     mean = dataset.mean
    #     std = dataset.std

    # train_transforms = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.CLAHE(p=0.3),
    #     A.Resize(width, height),
    #     A.Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ])
    # valid_transforms = A.Compose([
    #     A.Resize(width, height),
    #     A.Normalize(mean=mean, std=std),
    #     ToTensorV2()
    # ])

    # dataset.set_transform(train_transforms, valid_transforms)
    # train_data, val_data = dataset.split_dataset()

    train_loader = BaseDataLoader(
        dataset,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True
    )

    # val_loader = DataLoader(
    #     val_data,
    #     batch_size=configs['train']['batch_size'],
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=False
    # )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(configs['model']).to(device)

    loss_fn = get_loss(configs['train']['loss'])
    optimizer = get_optim(configs['train']['optim'], model, configs)
    scheduler = None

    save_dir = os.path.join(configs['ckpt_path'], str(model.name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    i = 0
    while True:
        version = 'v' + str(i)
        if os.path.exists(os.path.join(save_dir, version)):
            if not os.listdir(os.path.join(save_dir, version)):
                save_dir = os.path.join(save_dir, version)
                break
            else:
                i += 1
        else:
            save_dir = os.path.join(save_dir, version)
            os.makedirs(save_dir)
            break

    best_loss = 100
    cnt = 0
    epoch = configs['train']['epoch'] if not configs['fast_train_mode'] else 1
    for e in range(epoch):
        print(f'Epoch {e+1}\n-------------------------------')
        train(
            configs, train_loader, device, model, loss_fn,
            optimizer, scheduler, e+1, #configs['train']['mix']
        )
        # val_loss = validation(
        #     val_loader, save_dir, device, model, loss_fn, e+1
        # )
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     cnt = 0
        # else:
        #     cnt += 1
        # if cnt == configs['train']['early_patience']:
        #     print('Early Stopping!')
        #     break
        print('\n')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="/workspace/data/train.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    seed_everything(configs['seed'])
    run_pytorch(configs=configs)
