import os
import sys
import glob
import pprint
from pathlib import Path
from collections import defaultdict
import tqdm
import joblib
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn
from PIL import Image, ImageDraw


def save_image_sizes():
    # Save image sizes to pickle
    # os.path.splitext(os.path.basename("/path/to/file.txt"))[0]
    image_sizes = defaultdict(tuple)
    for image_path in tqdm.tqdm(glob.glob("/path/to/test/images/*.jpg")):  # 경로 설정
        with open(image_path, 'rb') as file:
            image = Image.open(file).convert("RGB")
            # w, h = image.size
            image_sizes[Path(image_path).stem] = image.size

    joblib.dump(image_sizes, "image_sizes.pkl")

image_sizes = joblib.load("image_sizes.pkl")


nonlabels = defaultdict(int)
nonlabel_files = defaultdict(list)


def ensemble_wbf_boxes():
    pred_dirs = [
        "datasets/results/tld_yolov11x_d2_2",
       "datasets/results/tld_yolov11x_d1_cos",
       "datasets/results/co_detr_3e_nms_0.001",
       "datasets/results/dino_swin_4e_nms_0.001",
    ]

    col_names = ["class_id", "norm_center_x", "norm_center_y", "norm_w", "norm_h", "conf_score"]
    for pred_path in tqdm.tqdm(sorted(glob.glob("/workspace/traffic_light/data/detection/test/images/*.jpg"))):  # 경로 설정
        boxes = []
        scores = []
        labels = []

        for pred_dir in pred_dirs:
            filename = f"{Path(pred_path).stem}.txt"
            if not os.path.isfile(os.path.join(pred_dir, filename)):
                continue

            df = pd.read_csv(os.path.join(pred_dir, filename), sep=" ", names=col_names)
            if not len(df):
                nonlabels[pred_dir] += 1
                nonlabel_files[pred_dir].append(filename)

            w, h = image_sizes[Path(pred_path).stem]

            xyxy = xywhn2xyxy(df.iloc[:, 1:-1].values, w=w, h=h)
            xyxy[:, [0, 2]], xyxy[:, [1, 3]] = xyxy[:, [0, 2]] / w, xyxy[:, [1, 3]] / h

            boxes.append(xyxy)
            scores.append(df.iloc[:, -1].values)
            labels.append(df.iloc[:, 0].values)

        boxes, scores, labels = \
            weighted_boxes_fusion(boxes, scores, labels, weights=[1, 1, 2, 1], iou_thr=0.25, skip_box_thr=0)

        boxes[:, [0, 2]], boxes[:, [1, 3]] = boxes[:, [0, 2]] * w, boxes[:, [1, 3]] * h
        boxes = xyxy2xywhn(boxes, w=w, h=h).round(7)
        scores = scores.round(7)

        os.makedirs('./pred_ensemble', exist_ok=True)  # 경로 설정 
        with open(os.path.join("./pred_ensemble", filename), 'w') as f:  # 경로 설정
            if not len(boxes):
                nonlabels["pred_ensemble"] += 1
                nonlabel_files["pred_ensemble"].append(filename)

            for b, s, l in zip(boxes, scores, labels):
                if s < 0.0:
                    continue
                f.write(f"{int(l)} {b[0]} {b[1]} {b[2]} {b[3]} {s}\n")

    print(nonlabels)
    pprint.pprint(nonlabel_files)

def visualize_ensemble():
    categories = categories = [
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
        {"id": 13, "name": "bus_warning", "supercategory": "bus_warning"},
    ]

    for label_path in tqdm.tqdm(sorted(glob.glob("./pred_ensemble/*.txt"))):
        img = Image.open(os.path.join("/workspace/traffic_light/data/detection/test/images", f"{Path(label_path).stem}.jpg")).convert("RGB")
        original_w, original_h = img.size
        # print(original_w, original_h)
        draw = ImageDraw.Draw(img)

        categories = [
            "veh_go",
            "veh_goLeft",
            "veh_noSign",
            "veh_stop",
            "veh_stopLeft",
            "veh_stopWarning",
            "veh_warning",
            "ped_go",
            "ped_noSign",
            "ped_stop",
            "bus_go",
            "bus_noSign",
            "bus_stop",
            "bus_warning",
        ]
        id2label = {index: x for index, x in enumerate(categories, start=0)}
        label2id = {v: k for k, v in id2label.items()}

        with open(label_path, "r") as file:
            for line in file:
                class_id, norm_center_x, norm_center_y, norm_w, norm_h, cs = \
                    map(float, line.strip().split())

                x_min = (norm_center_x - norm_w / 2) * original_w
                y_min = (norm_center_y - norm_h / 2) * original_h
                x_max = (norm_center_x - norm_w / 2) * original_w + norm_w * original_w
                y_max = (norm_center_y - norm_h / 2) * original_h + norm_h * original_h

                draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=1)
                draw.text(
                    (x_min, y_min), 
                    f"{id2label[class_id]}({round(cs, 2)})",
                    fill="white", 
                    stroke_width=1, 
                    stroke_fill="black"
                )
        img_out_path='./pred_ensemble_images/'
        os.makedirs(img_out_path, exist_ok=True)
        img.save(os.path.join(img_out_path, f"{Path(label_path).stem}.jpg"))


if __name__ == "__main__":
    # save_image_sizes()
    ensemble_wbf_boxes()
    # visualize_ensemble()
