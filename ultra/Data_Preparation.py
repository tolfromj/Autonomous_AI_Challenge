import os
import shutil
import uuid
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

import argparse

iou_scores = []

def create_new_data_directory(original_data_dir, new_data_dir, split_ratio=0.8):
    # Ensure the new data directory exists
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)

    # Define paths for images and labels
    new_images_dir = os.path.join(new_data_dir, 'images')
    new_labels_dir = os.path.join(new_data_dir, 'labels')

    # Ensure the new directories for images and labels exist
    for subdir in ['train']:
        os.makedirs(os.path.join(new_images_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(new_labels_dir, subdir), exist_ok=True)

    # List all image files and corresponding label files for train and val sets
    def list_files(base_dir, split_type):
        image_files = []
        label_files = []
        for cityname in os.listdir(base_dir):
            img_dir = os.path.join(base_dir, cityname, 'img')
            label_dir = os.path.join(base_dir, cityname, 'new_txt')
            if os.path.exists(img_dir) and os.path.exists(label_dir):
                for img_file in os.listdir(img_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(img_dir, img_file))
                        label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
                        if os.path.exists(label_file):
                            label_files.append(label_file)
        return list(zip(image_files, label_files))

    train_files = list_files(os.path.join(original_data_dir, 'train'), 'train')

    # Copy files and rename them uniquely
    def copy_and_rename_files(files, split_type):
        txt_file_path = os.path.join(new_data_dir, f'{split_type}.txt')
        with open(txt_file_path, 'w') as txt_file:
            for image_path, label_path in files:
                #unique_id = uuid.uuid4().hex
                unique_id = os.path.splitext(image_path)[0].split('/')[-3]+'_'+os.path.splitext(image_path)[0].split('/')[-1]
                
                # Copy image file
                img_ext = os.path.splitext(image_path)[1]
                new_image_name = f'{unique_id}{img_ext}'
                dest_image_path = os.path.join(new_images_dir, split_type, new_image_name)
                shutil.copy(image_path, dest_image_path)

                # Copy and reformat label file
                new_label_name = f'{unique_id}.txt'
                dest_label_path = os.path.join(new_labels_dir, split_type, new_label_name)
                reformat_label_file(label_path, dest_label_path, image_path)

                # Write new image path to txt file
                txt_file.write(dest_image_path + '\n')

    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0
    def recreate_mask_from_polygons(polygons, size):
        mask = np.zeros(size, dtype=np.uint8)
        for polygon in polygons:
            contour = np.array(polygon).reshape((-1, 2))
            cv2.fillPoly(mask, [contour], 255)
        return mask
    def remove_noise(mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    # Reformat the label file to include instance segmentation mask coordinates
    def reformat_label_file(src_label_path, dest_label_path, image_path):
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        mask_dir = os.path.join(os.path.dirname(image_path), '..', 'instance')
        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        with open(src_label_path, 'r') as src_file:
            lines = src_file.readlines()
        with open(dest_label_path, 'w') as dest_file:
            line_num = 0
            writen_line = 0
            for line in lines:
                line_num+=1
                parts = line.strip().split()
                if len(parts) >= 10:
                    x1, y1, x2, y2, class_id, loc_id, brake, incatlft, incatrht, hazlit = map(float, parts[:10])

                    # Extract the mask for the current instance
                    instance_mask = np.zeros_like(mask)
                    instance_mask[mask == int(line_num)] = 255
                    new_instance_mask = remove_noise(instance_mask)
                    
                    polygons = mask2polygon(new_instance_mask,cv2.CHAIN_APPROX_NONE)
                    len_polygons=0
                    for pp in polygons:
                        len_polygons += len(pp)
                    '''
                    if len_polygons>200:
                        # Use the provided mask2polygon method to convert mask to polygons
                        polygons = mask2polygon(new_instance_mask)
                    '''
                    
                    # Recreate mask from polygons
                    recreated_mask = recreate_mask_from_polygons(polygons, instance_mask.shape)
                    
                    # Calculate IoU between original and recreated mask
                    iou_score = calculate_iou(instance_mask, recreated_mask)
                    iou_scores.append(iou_score)
                    if iou_score>0.01:
                        normalized_contours = []
                        for polygon in polygons:
                            normalized_contours.extend([(polygon[i] / width if i % 2 == 0 else polygon[i] / height) for i in range(len(polygon))])
                        if len(normalized_contours)==0:
                            pass
                        else:
                            # Write to destination label file in the desired format
                            new_label_line = f'{int(class_id)} {int(loc_id)} {int(brake)} {int(incatlft)} {int(incatrht)} {int(hazlit)} ' + ' '.join(map(str, normalized_contours))
                            dest_file.write(new_label_line + '\n')
                            writen_line+=1
                else:
                    print(parts)
    # Process train and val sets

    copy_and_rename_files(train_files, 'train')

# Use the mask2polygon function provided in the initial code
def mask2polygon(image, mode=cv2.CHAIN_APPROX_TC89_KCOS ):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, mode)
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons 

# Use the provided helper functions for merging contours
def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0

def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = (p2[0][0] - p1[0][0]) ** 2 + (p2[0][1] - p1[0][1]) ** 2
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in range(0, idx1 + 1):
        contour.append(contour1[i])
    for i in range(idx2, len(contour2)):
        contour.append(contour2[i])
    for i in range(0, idx2 + 1):
        contour.append(contour2[i])
    for i in range(idx1, len(contour1)):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour

def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_data_dir", type=str, default="../data/segmantation"
    )
    parser.add_argument(
        "--new_data_dir", type=str, default="../data/segmantation_coco"
    )
    args = parser.parse_args()
    
    
    create_new_data_directory(args.original_data_dir, args.new_data_dir, split_ratio=0.8)
    
    avg_iou = sum(iou_scores) / len(iou_scores)
    print(f"Average IoU between img vs polygon: {avg_iou}")



