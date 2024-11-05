import os
import pickle

import numpy as np
from tqdm import tqdm


def nms(bounding_boxes, confidence_score, labels, threshold):
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
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_labels = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

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
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_labels

def xyxy2xywhn(bbox, w, h):
    x_min, y_min, x_max, y_max  = bbox
    if x_max > w:
        x_max = float(w)
    if x_min < 0:
        x_min = float(0)
    if y_max > h:
        y_max = float(h)
    if y_min < 0:
        y_min = float(0)
    norm_w = np.abs(x_max-x_min)/w
    norm_h = np.abs(y_max-y_min)/h
    norm_center_x = (x_max+x_min)/(2*w)
    norm_center_y = (y_max+y_min)/(2*h)
    return (norm_center_x, norm_center_y, norm_w, norm_h)

def pkl2txt(cpkt_path, output_label_path, nms_iou_threshold=0.001, score_threshold=0.0):
    empty_file_cnt=0
    os.makedirs(output_label_path, exist_ok=True)
    with open(cpkt_path, 'rb') as f:
        labels=pickle.load(f)

        for label in tqdm(labels):
            lines=[]
            file_num = label['img_id']
            original_h, original_w = label['ori_shape']
            picked_boxes, picked_score, picked_categories = nms(label['pred_instances']['bboxes'], 
                                            label['pred_instances']['scores'],
                                            label['pred_instances']['labels'], 
                                            nms_iou_threshold)
            
            for bbox, score, category in zip(picked_boxes, picked_score, picked_categories):
                if score < score_threshold:
                    continue
                category = category.item()
                x_min, y_min, x_max, y_max  = bbox.numpy()
                norm_center_x, norm_center_y, norm_w, norm_h  = xyxy2xywhn((x_min, y_min, x_max, y_max), original_w, original_h)
                
                # remove too small and too big boxes 
                if (norm_h < 0.007 or norm_h > 0.3) and score < 0.2:
                    continue
                if (norm_w < 0.007 or norm_w > 0.3) and score < 0.2:
                    continue
                
                lines.append((category, norm_center_x, norm_center_y, norm_w, norm_h, score))
            
            if not lines:
                empty_file_cnt+=1
                print(f'{file_num}.txt')

            with open(os.path.join(output_label_path, f'{file_num}.txt'), 'w') as file:
                for line in lines:
                    category, norm_center_x, norm_center_y, norm_w, norm_h, score = line
                    file.write("%d %lf %lf %lf %lf %lf\n"
                    % (category, norm_center_x, norm_center_y, norm_w, norm_h, score))      

    print("no box files: ", empty_file_cnt)
    

if __name__=="__main__":
    cpkt_path =''  # path/to/pkl file. e.g) outputs/3_epoch_co_detr.pkl
    output_label_path=''  # path/to/result/folder name. e.g) results/co_detr_3e_nms_0.001

    nms_iou_threshold=0.001
    score_threshold=0.0

    pkl2txt(cpkt_path=cpkt_path, output_label_path=output_label_path, nms_iou_threshold=nms_iou_threshold, score_threshold=score_threshold)
    print("done")
