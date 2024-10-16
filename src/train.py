
import os
import multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import albumentations as A

from models import get_model
from datasets import TrafficLightDataset
from utils.collate_fn import get_collate
from torch.utils.data import DataLoader

# def train() -> None:
def train(
    dataloader: DataLoader, device: str,
    model: nn.Module, #optimizer: _Optimizer, scheduler: _Scheduler,
    epoch: int
) -> None:
    
    # for name, layer in model.namesd_modules():
    #     print(name, layer)
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.class_labels_classifier.parameters():
        param.requires_grad = True

    for param in model.bbox_predictor.parameters():
        param.requires_grad = True
    
    model.train()

    train_loss = 0

    for batch_idx, (images, targets) in tqdm(enumerate((dataloader))):
        
        images = images.to(device)
        if type(targets) == dict: # detr에 경우.
            targets["pixel_mask"] = targets["pixel_mask"].to(device)
            for idx, labels in enumerate(targets["labels"]):
                for label,value in labels.items():
                    targets["labels"][idx][label] = value.to(device)

        else: targets = targets.to(device)


        model.optimizer.zero_grad()
        outputs = model(images, targets)
        loss = outputs["loss"]
        loss.backward()
        model.optimizer.step()
        val_loss = loss.item()
        log_term = 1# len(dataloader) // 5
        if (batch_idx+1) % log_term == 0:
            train_loss = val_loss / log_term

            print(
                f"Epoch[{epoch}]({batch_idx + 1}/{len(dataloader)}) "
                f"| lr {model.learning_rate} \ntrain loss {train_loss:4.4}"
            )

        break
            # lr_decay check
            # if epoch == 9:
            #     lr_decay = 0.1
            #     model.scale_lr(lr_decay) 
            #     model.learning_rate  = model.learning_rate * lr_decay

            # grad check
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"After zero_grad, {name} grad: {param.grad}")


            # print(f"epoch_{batch_idx}: loss: {return_dict["loss"]}")

def validation(
    dataloader: DataLoader, save_dir: os.PathLike,
    model: nn.Module,  device: str, epoch: int
) -> float:
    model.eval()
    valid_loss = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            
            # setting device
            images = images.to(device)
            if "pixel_mask" in targets.keys(): # detr에 경우.
                targets["pixel_mask"] = targets["pixel_mask"].to(device)
                for idx, labels in enumerate(targets["labels"]):
                    for label,value in labels.items():
                        targets["labels"][idx][label] = value.to(device)

            outputs = model(images, targets)
            loss = outputs["loss"] # loss : torch.Tensor

            valid_loss.append(loss.cpu().numpy())

    val_loss = np.sum(valid_loss)/len(dataloader)
    

    print(
        f"Epoch[{epoch}]({len(dataloader)})"
        f"valid loss {val_loss:4.4}"
    )

    torch.save(
        model.state_dict(),
        f'{save_dir}/{epoch}-{val_loss:4.4}.pth'
        )
    print(
        f'Saved Model to {save_dir}/{epoch}-{val_loss:4.4}.pth'
    )
    return val_loss

def run_pytorch(root_dir, model_name, ckpt_path, batch_size=1, epoch = 1, early_patience = 5) -> None:
    """
    학습 파이토치 파이프라인

    :param configs: 학습에 사용할 config
    :type configs: dict
    """

    dataset = TrafficLightDataset(
        root_dir, # root_dir; "/workspace/traffic_light/data/detection/train/"
    )
    collate_fn = get_collate(model_name)
    train_augments_for_huggingface = A.Compose([
        A.RandomScale(scale_limit=0.2, p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.Resize(height=480, width=480, p=1.0),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    val_augments_for_huggingface = A.Compose([
        A.RandomScale(scale_limit=0.2, p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.Resize(height=480, width=480, p=1.0),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    train_loader = DataLoader(
        dataset(train_augments_for_huggingface),
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset(val_augments_for_huggingface),
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name).to(device) # model_name : "facebook/detr-resnet-50"

    save_dir = os.path.join(ckpt_path, str(model_name))
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

    epoch: int
    best_loss = 100
    cnt = 0

    for e in range(epoch):
        print(f'Epoch {i+1}\n-------------------------------')
        train(
            train_loader, device, model, #optimizer, scheduler,
            e+1
        )
        val_loss = validation(
            val_loader, save_dir, model, device, e+1
        )
        if val_loss < best_loss:
            best_loss = val_loss
            cnt = 0
        else:
            cnt += 1
        if cnt == early_patience:
            print('Early Stopping!')
            break
        print('\n')
    print('Done!')

if __name__ == '__main__':
    root_dir = "/workspace/traffic_light/data/detection/train/"
    model_name = "facebook/detr-resnet-50"
    ckpt_path = "/workspace/traffic_light/output"
    batch_size = 4
    epoch = 1
    early_patience = 5
    run_pytorch(root_dir, model_name, ckpt_path, batch_size, epoch, early_patience)






    # TEST_AUGMENTS = A.Compose([
    #     A.Resize(height=480, width=480, p=1.0),
    #     ToTensorV2(p=1.0)
    # ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    # TRAIN_AUGMENTS_FOR_ULTRALYTICS = A.Compose([
    #     # 스케일을 무작위로 변경하는 증강
    #     A.RandomScale(scale_limit=0.2, p=1.0),
    #     A.RandomBrightnessContrast(p=1.0),
    #     # 이미지를 512x512 크기로 리사이징
    #     A.Resize(height=480, width=480, p=1.0),
    #     # PyTorch 모델에 입력하기 위한 변환
    #     # ToTensorV2(p=1.0),
    #     A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, p=1.0)
    # ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))