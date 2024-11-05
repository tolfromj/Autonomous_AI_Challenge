import os
import glob
import argparse

import torch
import numpy as np
import ttach as tta
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension.lstrip(".")


def main(model_filename, input_test_path, output_path):
    result_path = output_path
    test_db_path = input_test_path
    test_res_path = "%s/predictions" % (result_path)

    if not os.path.exists(test_res_path):
        os.makedirs(test_res_path)

    img_exts = ["jpg", "bmp", "png"]
    img_files = list()
    for img_ext in img_exts:
        img_files += glob.glob("%s/images/*.%s" % (test_db_path, img_ext))

    img_files.sort()

    model = YOLO(model_filename)
    
    augmentations = tta.Compose(
    [
        # tta.HorizontalFlip(),
        # tta.Scale(scales=[1, 2, 4]),
        tta.Resize(sizes=[800,800]),
        tta.Multiply(factors=[0.9, 1, 1.3]),        
    ]
    )
    transform = transforms.ToTensor()
    # tta_model = tta.SegmentationTTAWrapper(model, transforms)
    # img_filename -> ../../data/detection/test/images/00001234.jpg
    for idx, img_filename in enumerate(img_files):
        if 8652 > idx or idx > 8653:
            continue
        img = transform(Image.open(img_filename)).unsqueeze(0)
        print(img.shape)
        # break
        bboxes =[]
        for index, augmentation in enumerate(augmentations):
            augmented_img = augmentation.augment_image(img)
            result = model.predict(augmented_img, imgsz=1280, conf=0.001, iou=0.6, device='cpu')[0]
        # result = model.predict(img_filename, imgsz=1280, conf=0.001, iou=0.6, augment=True)[0]
        # result = model.predict(img_filename)[0]
            deaug_boxes = augmentation.deaugment_label(result.boxes)
            # print(type(deaug_boxes))
            bboxes.append(deaug_boxes)
            # print(deaug_boxes)
            # break
            if index == 20:
                break
        # boxes = 
        print("bboxes len: ",len(bboxes))
        # break
        img_ext = get_file_extension(img_filename)
        txt_filename = img_filename.replace(
            img_ext, "txt"
        )  # ../../data/detection/test/images/00001234.txt
        txt_filename = txt_filename.replace(
            "../data/detection/test/images", "Result/detect/predictions"
        )  # ../../data/detection/test/predictions/00001234.txt
        # boxes = result.boxes  # => ../Result/detect/predictions
        # num_obj = len(boxes.cls)

        with open(txt_filename, "w") as f1:
            for boxes in bboxes:
                num_obj = len(boxes.cls)
                for obj_idx in range(num_obj):
                    cls_id = int(boxes.cls[obj_idx])
                    cs = boxes.conf[obj_idx]
                    xywhn = boxes.xywhn[obj_idx]
                    # class_id norm_center_x norm_center_y norm_w norm_h confidence_score
                    f1.write(
                        "%d %lf %lf %lf %lf %lf\n"
                        % (cls_id, xywhn[0], xywhn[1], xywhn[2], xywhn[3], cs)
                    )

                # xywh = boxes.xywh[obj_idx]
                # f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, cs, xywh[0], xywh[1],xywh[2],xywh[3]))

        # if num_obj == 0:
        #     print(txt_filename)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/workspace/traffic_light/ultra/output/runs/detect/traffic_yolov11_s_sgd/weights/best.pt",
    )
    parser.add_argument(
        "--input_test_path",
        type=str,
        default="../../data/detection/test"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../Result/detect"
    )

    args = parser.parse_args()
    main(args.model_ckpt, args.input_test_path, args.output_path)  # 절대경로로 변경할 것.
