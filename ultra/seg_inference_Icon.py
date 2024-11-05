import numpy as np
import glob, cv2
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

# import albumentations.pytorch
import torch
import torch.distributed as dist
import torch.nn as nn

import torchvision
import os
import random
import torch.utils.data as data_utils
import datetime
from shutil import copyfile
import time
import errno

import scipy
import torch.nn.functional as F


from ultralytics import YOLO


model = YOLO("../output/runs/segment/segmentation_yolov11_s_sgd/weights/best.pt").cuda()
print(model.task)


print(model.model.end2end)


# --------------Class Param------------
agent_classes = ["Car", "Bus"]
loc_classes = ["VehLane", "OutgoLane", "IncomLane", "Jun", "Parking"]
action_classes = ["Brake", "IncatLft", "IncatRht", "HazLit"]
class_nums = [len(agent_classes), len(loc_classes), len(action_classes)]
# --------------Class Param------------


icons = {}
for actions in action_classes:
    target = "./Icons/" + actions + ".png"
    icon_img = cv2.imread(target)
    icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
    icons[actions] = icon_img

for actions in loc_classes:
    # print(actions)
    target = "./Icons/" + actions + ".png"
    icon_img = cv2.imread(target)
    icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
    icons[actions] = icon_img


def seg_plot_one_box(
    x,
    idx,
    img,
    mask,
    cls,
    loc,
    action,
    color=None,
    label=None,
    track_id=None,
    line_thickness=None,
):
    # Plots one bounding box on image img
    tl = line_thickness or 2  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    c1, c2 = (
        np.clip(int(x[0]), 0, img.shape[1]),
        np.clip(int(x[1]), 0, img.shape[0]),
    ), (np.clip(int(x[2]), 0, img.shape[1]), np.clip(int(x[3]), 0, img.shape[0]))

    cv2.rectangle(img, c1, c2, color, thickness=1)

    agent_list = ["Car", "Bus"]
    loc_list = ["VehLane", "OutgoLane", "IncomLane", "Jun", "Parking"]
    action_list = ["Brake", "IncatLft", "IncatRht", "HazLit"]

    num_icon = np.sum(action)

    icon_size = int(np.min([(c2[0] - c1[0]) / num_icon, (x[3] - x[1]) / 2, 64]))
    c3 = c1[0]  # +(c2[0]-c1[0])//2-icon_size*num_icon//2

    try:
        offset_icon = 0
        for ii in range(len(action)):
            if action[ii] == 1:

                img[
                    c1[1] : c1[1] + icon_size,
                    c3 + offset_icon : c3 + offset_icon + icon_size,
                    :,
                ] = (
                    cv2.resize(
                        icons[action_list[ii]],
                        (icon_size, icon_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    * 0.5
                    + img[
                        c1[1] : c1[1] + icon_size,
                        c3 + offset_icon : c3 + offset_icon + icon_size,
                        :,
                    ]
                    * 0.5
                )
                offset_icon += icon_size

        img[c2[1] - icon_size : c2[1], c3 : c3 + icon_size, :] = (
            cv2.resize(icons[loc_list[loc]], (icon_size, icon_size)) * 0.5
            + img[c2[1] - icon_size : c2[1], c3 : c3 + icon_size, :] * 0.5
        )

    except:
        pass

    # Expand mask dimensions to match the image
    mask = mask[c1[1] : c2[1], c1[0] : c2[0]]
    mask = mask > 0.5

    img[c1[1] : c2[1], c1[0] : c2[0], :][mask] = (
        img[c1[1] : c2[1], c1[0] : c2[0], :][mask] * 0.65 + np.array(color) * 0.35
    )


import shutil
import time

try:
    shutil.rmtree("./Result/")
    print("Removed New dir")
except:
    print("Making New dir")

filepath = "./Result/"
if not os.path.exists(os.path.dirname(filepath)):
    try:
        os.makedirs(os.path.dirname(filepath))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

COLORS = [
    [255, 0, 0],  # Red
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
    [255, 255, 0],  # Yellow
    [255, 0, 255],  # Magenta
    [0, 255, 255],  # Cyan
    [128, 0, 0],  # Maroon
    [0, 128, 0],  # Green (dark)
    [0, 0, 128],  # Navy
    [128, 128, 0],  # Olive
    [128, 0, 128],  # Purple
    [0, 128, 128],  # Teal
    [255, 165, 0],  # Orange
    [210, 180, 140],  # Tan
    [255, 192, 203],  # Pink
    [0, 128, 128],  # Teal
    [255, 99, 71],  # Tomato
    [139, 69, 19],  # Saddle Brown
    [0, 128, 0],  # Green (dark)
    [255, 20, 147],  # Deep Pink
]


count = 0
frame_num = 0
write = False
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
old_reid_feat = []
reid_feat = []
track_num = 1
track_thresh = 0.5
target_folder = "/workspace/traffic_light/data/segmentation/test/IllegalParking01_Rain"  # Target Dir

try:
    searchLabel = sorted(os.listdir(target_folder))
except:
    print("Invalid Target Dir")

with torch.no_grad():
    for jj in range(len(searchLabel) - 1):
        if jj % 70 == 0 and jj < 1000:

            # ===============================
            t1.append(time.time())
            # ===============================

            img_name = target_folder + "/" + searchLabel[jj]
            results = model(img_name, verbose=False, imgsz=[480, 1280])

            # ===============================
            t2.append(time.time())
            # ===============================
            target_outputs = results[0].boxes.data.cpu().numpy()
            target_img = results[0].orig_img

            target_img = np.array(target_img[:, :, ::-1])

            xyxy = target_outputs[:, 0:4]
            cls = target_outputs[:, 5].astype("int")
            loc = target_outputs[:, 6].astype("int")
            action = target_outputs[:, 7:].astype("int")
            # ===============================
            t3.append(time.time())
            # ===============================
            for i in range(xyxy.shape[0]):
                # print(naming_num[i])

                seg_plot_one_box(
                    xyxy[i],
                    i,
                    target_img,
                    results[0].masks.data[i].cpu().numpy(),
                    cls[i],
                    loc[i],
                    action[i],
                    color=COLORS[i % len(COLORS)],
                )

            # ===============================

            t4.append(time.time())

            if write:
                target_img = target_img.copy()

                path = "./Result/" + str(jj).zfill(6) + ".png"
                cv2.imwrite(path, target_img[:, :, ::-1])
            else:
                plt.rcParams["figure.figsize"] = [20, 10]
                plt.imshow(target_img)
                plt.show()


print("Run model              :", np.median((np.array(t2) - np.array(t1))))
print("Post Process           :", np.median((np.array(t3) - np.array(t2))))
print("Plot Results           :", np.median((np.array(t4) - np.array(t3))))
