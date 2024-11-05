# How To Train Object Detection and Segmentation

## 0. install 
### 0.1. Install ultralytics
```
pip install ultralytics
````

### 0.2. Create conda env.
```
conda create -n ultra python=3.8 -y
conda activate ultra
cd ultralytics
pip install -e .
```

# Object Detection for 신호등 인식
## 1. Split the dataset into training and validation sets.
```
# setting data directory

# datasets
# └── data_traffic   
#     ├── test
#     └── train
#         ├── images
#         |   ├── 00000000.jpg
#         |   └── 00000001.jpg ...
#         └── labels
#             ├── 00000000.txt
#             └── 00000001.txt ...

python split_3_dataset_traffic_light.py --data_path ../datasets/data_traffic
```

## 2. Edit dataset root dir
- /ultralytics/datasets/traffic_light.yaml
- path: ../datasets/data_traffic


## 3. Train
Before train, Download pre-trained model . [YOLO11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)

```
cd ultra/tests
python train_yolo11.py
```

## 4. validation
- set path of model_filename
```
cd ultra/tests
python val_yolo11.py
```

## 5. test
```
cd ultra/tests 
python test_yolo11.py --model_ckpt ../output/runs/detect/traffic_yolov11_s_sgd/weights/best.pt
```  
