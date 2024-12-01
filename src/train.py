import os
import argparse
import multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A

from models import get_model
from datasets import TrafficLightDataset
from utils.collate_fn import get_collate
from utils.compute_metrics import compute_metrics
import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)  # Python random seed 설정
    np.random.seed(seed)  # NumPy random seed 설정
    torch.manual_seed(seed)  # PyTorch CPU seed 설정
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed 설정
    torch.cuda.manual_seed_all(seed)  # 여러 GPU를 사용하는 경우 모두 설정
    torch.backends.cudnn.deterministic = True  # 연산의 결정론적 동작 설정
    torch.backends.cudnn.benchmark = False  # 성능을 약간 희생하고 완벽한 재현성을 보장

# def train() -> None:
def train(
    dataloader: DataLoader,
    device: str,
    model: nn.Module,  # optimizer: _Optimizer, scheduler: _Scheduler,
    epoch: int,
) -> None:

    # for name, layer in model.namesd_modules():
    #     print(name, layer)

    model.train()

    train_loss = 0

    for batch_idx, (images, targets) in enumerate(tqdm((dataloader))):
        images = images.to(device)
        if type(targets) is dict:  # detr에 경우.
            targets["pixel_mask"] = targets["pixel_mask"].to(device)
            for idx, labels in enumerate(targets["labels"]):
                for label, value in labels.items():
                    targets["labels"][idx][label] = value.to(device)

        else:
            targets = targets.to(device)

        model.optimizer.zero_grad()
        outputs = model(images, targets)
        loss = outputs["loss"]
        loss.backward()
        model.optimizer.step()
        train_loss = loss.item()
        # log_term = 100  # len(dataloader) // 5
        # if (batch_idx + 1) % log_term == 0:
        # train_loss = val_loss / log_term
        # if batch_idx==31:
        #     break
    print(
        f"Epoch[{epoch}]({batch_idx + 1}/{len(dataloader)}) "
        f"| lr {model.learning_rate} \n train loss {train_loss:4.4}"
    )

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
    dataloader: DataLoader,
    save_dir: os.PathLike,
    model: nn.Module,
    device: str,
    epoch: int,
) -> float:
    model.eval()
    valid_loss = []

    print("start validataion ...")
    metrics={}
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
            # if batch_idx == 100:
            #     break
            # setting device
            images = images.to(device)
            if "pixel_mask" in targets.keys():  # detr에 경우.
                targets["pixel_mask"] = targets["pixel_mask"].to(device)
                for idx, labels in enumerate(targets["labels"]):
                    for label, value in labels.items():
                        targets["labels"][idx][label] = value.to(device)

            outputs = model(images, targets)
            loss = outputs["loss"]  # loss : torch.Tensor

            logits = outputs["logits"]
            pred_boxes = outputs["pred_boxes"]
            
            if batch_idx == len(dataloader) - 1:
                metrics=compute_metrics(logits, pred_boxes, targets["labels"], compute_result=True)
            else: 
                metrics=compute_metrics(logits, pred_boxes, targets["labels"], compute_result=False)
            
            valid_loss.append(loss.cpu().numpy())
    print('mAP: ', metrics['mAP'], ' ',
                  'mAP_50: ', metrics['mAP_50'], ' ',
                  'mAP_75: ', metrics['mAP_75'])
    val_loss = np.sum(valid_loss) / len(dataloader)

    print(f"Epoch[{epoch}]({len(dataloader)})" f"valid loss {val_loss:4.4}")

    return val_loss


def run_pytorch(
    root_dir,
    model_name,
    output_dir,
    resume_from,
    batch_size=1,
    epoch=1,
    early_patience=5,
    cuda=0,
) -> None:
    """
    학습 파이토치 파이프라인

    :param configs: 학습에 사용할 config
    :type configs: dict
    """
    device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"

    if resume_from:
        model_data = torch.load(resume_from)
        model = get_model(model_name, device).to(device)
        model.load_state_dict(model_data["model_state_dict"])
        model.optimizer.load_state_dict(model_data["optimizer_state_dict"])
        start_epoch = model_data["epoch"] + 1

    else:
        model = get_model(model_name, device).to(
            device
        )  # model_name : "facebook/detr-resnet-50"
        start_epoch = 0

    train_dataset = TrafficLightDataset(
        root_dir,
        mode="train",
        image_processor=model_name,  # root_dir; "/workspace/traffic_light/data/detection/"
    )
    val_dataset = TrafficLightDataset(root_dir, mode="val", image_processor=model_name)
    collate_fn = get_collate(model_name)
    train_augments_for_huggingface = A.Compose(
        [
            # A.RandomScale(scale_limit=0.2, p=1.0),
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(height=800, width=800, p=1.0),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )

    val_augments_for_huggingface = A.Compose(
        [
            # A.RandomScale(scale_limit=0.2, p=1.0),
            # A.RandomBrightnessContrast(p=1.0),
            A.Resize(height=800, width=800, p=1.0),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )

    train_loader = DataLoader(
        train_dataset(train_augments_for_huggingface),
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count() // 4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset(val_augments_for_huggingface),
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count() // 4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # pre-trained model freezing
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.class_labels_classifier.parameters():
    #     param.requires_grad = True
    # for param in model.bbox_predictor.parameters():
    #     param.requires_grad = True

    save_dir = os.path.join(output_dir, str(model_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0
    while True:
        version = "v" + str(i)
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
    best_loss = 1e10000
    cnt = 0

    for e in range(start_epoch, epoch):
        print(f"Epoch {e+1}\n-------------------------------")
        train(train_loader, device, model, e + 1)  # optimizer, scheduler,
        val_loss = validation(val_loader, save_dir, model, device, e + 1)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "loss": val_loss,
                },
                f"{save_dir}/best_{e}.pth",
            )
            print(
                f"Saved best_loss Model to {save_dir}/best.pth ({e}/{epoch}-{val_loss:4.4})"
            )
            cnt = 0
        else:
            cnt += 1
        if cnt == early_patience:
            print("Early Stopping!")
            break
        print("\n")
    print("Done!")


if __name__ == "__main__":
    set_random_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/detr-resnet-101")
    parser.add_argument(
        "--root_dir", type=str, default="/workspace/traffic_light/data/detection/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/traffic_light/output"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,  # "/workspace/traffic_light/output/facebook/detr-resnet-101/v2/best.pth"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=5)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    run_pytorch(
        args.root_dir,
        args.model_name,
        args.output_dir,
        args.resume_from,
        args.batch_size,
        args.epoch,
        args.early_patience,
        args.cuda,
    )
    # val_augments_for_huggingface = A.Compose([
    #     A.RandomScale(scale_limit=0.2, p=1.0),
    #     A.RandomBrightnessContrast(p=1.0),
    #     A.Resize(height=480, width=480, p=1.0),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0)
    # ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # data_loader = DataLoader(TrafficLightDataset(root_dir, val_augments_for_huggingface), batch_size=4, shuffle=False, )#collate_fn=get_collate(model_name))#lambda x: tuple(zip(*x)))
    # print(next(iter(data_loader)))

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
