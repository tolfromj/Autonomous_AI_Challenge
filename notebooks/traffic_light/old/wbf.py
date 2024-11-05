import os
import sys

from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion as wbf

def nms(bounding_boxes, confidence_score, labels, orig_w, orig_h, threshold):
    """
    ref: https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
    the boxes format is xyxy
    e.g) 
    bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
    confidence_score = [0.9, 0.75, 0.8]
    threshold = 0.4
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = bounding_boxes
    # boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score=confidence_score
    # score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_labels = []

    # Compute areas of bounding boxes
    areas = ((end_x - start_x )*orig_w +1) * ((end_y - start_y)*orig_h+1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_labels.append(labels[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, (x2 - x1)* orig_w + 1)
        h = np.maximum(0.0, (y2 - y1)* orig_h + 1)

        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_labels

def xywhn2xyxyn(norm_center_x, norm_center_y, norm_w, norm_h):
    norm_center_x = float(norm_center_x)
    norm_center_y = float(norm_center_y)
    norm_w = float(norm_w)
    norm_h = float(norm_h)
    x_min = float(0) if (norm_center_x - norm_w / 2) < 0 else (norm_center_x - norm_w / 2)
    y_min = float(0) if (norm_center_y - norm_h / 2) < 0 else (norm_center_y - norm_h / 2) 
    x_max = float(1) if (norm_center_x + norm_w / 2) > 1 else (norm_center_x + norm_w / 2) 
    y_max = float(1) if (norm_center_y + norm_h / 2) > 1 else (norm_center_y + norm_h / 2) 
    return (x_min, y_min, x_max, y_max)

# def xyxy2xyxyn(bbox,w,h):
#     return (bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h)

def xyxyn2xywhn(bbox):
    x_min, y_min, x_max, y_max  = bbox
    norm_w = np.abs(x_max-x_min)
    norm_h = np.abs(y_max-y_min)
    norm_center_x = (x_max+x_min)/2
    norm_center_y = (y_max+y_min)/2
    return (norm_center_x, norm_center_y, norm_w, norm_h)

def make_df_list(results_path, image_dir):
    df_list = []
    file_names = []

    for i, result in enumerate(os.listdir(results_path)):
        data = []
        if i == 0:
            file_names.extend(sorted(os.listdir(os.path.join(results_path, result))))
        for file_name in tqdm(file_names):
            file_path = os.path.join(results_path, result, file_name)
            img_path = os.path.join(image_dir, file_name.replace('.txt','.jpg'))
            w, h = Image.open(img_path).size
            with open(file_path, 'r') as file:
                for line in file:
                    label, norm_center_x, norm_center_y, norm_w, norm_h, cs = line.split()
                    x_min, y_min, x_max, y_max = xywhn2xyxyn(norm_center_x, norm_center_y, norm_w, norm_h)
                    data.append([file_name, int(label), float(x_min), 
                                    float(y_min), float(x_max), float(y_max), float(cs), w, h])
        df_list.append(pd.DataFrame(data, columns=["file_name", "label", "x_min", 
                                        "y_min", "x_max", "y_max", "cs", "w", "h"]))
        
    print(f"{len(os.listdir(results_path))} dataframes were created")
    return df_list, file_names

def weights_box_fusion(results_path, image_dir, save_dir, nms_iou_threshold=0):
    
    df_list, file_names = make_df_list(results_path, image_dir)

    for file_name in tqdm(file_names):
        boxes_list=[]
        scores_list=[]
        labels_list=[]
        orig_w, orig_h = df_list[0][df_list[0]['file_name']==file_name][["w", "h"]].iloc[0]
        for df in df_list:
            sub_df = df[df['file_name']==file_name]
            if sub_df['cs'].values.size > 0:
                boxes_list.append(sub_df[['x_min', 'y_min', 'x_max', 'y_max']].values)
                scores_list.append(sub_df['cs'].values)
                labels_list.append(sub_df['label'].values)

        # if any(box_list for box_list in boxes_list if box_list):
        if any(box_list.size > 0 for box_list in boxes_list):
            boxes, scores, labels = wbf(
                boxes_list, scores_list, labels_list,
                weights=None, iou_thr=0.55, skip_box_thr=0.001
            )
            boxes, scores, labels = nms(
                boxes, scores, labels, orig_w, orig_h, nms_iou_threshold
            )

        with open(os.path.join(save_dir, file_name),'w') as f:
            for box, cs, label in zip(boxes, scores, labels):
                norm_center_x, norm_center_y, norm_w, norm_h=xyxyn2xywhn(box)
                f.write("%d %lf %lf %lf %lf %lf\n"%(label, norm_center_x, norm_center_y, norm_w, norm_h, cs))
    print('done')

if __name__=='__main__':
    results_path = '/workspace/traffic_light/submission/wbf_v4_yolo_dino_co'
    save_dir = '/workspace/traffic_light/submission/wbf_output/v5_nms(0.1)'
    image_dir = '/workspace/traffic_light/data/detection/test/images'
    nms_iou_threshold=0.1
    os.makedirs(save_dir, exist_ok=True)
    weights_box_fusion(results_path, image_dir, save_dir, nms_iou_threshold)