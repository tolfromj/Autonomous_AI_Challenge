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

# traffic_light
# ├── ultralytics
# |   ├── tests
# |   └── ultralytics
# |       └── datasets 
# └── data
#     └── detection  <- 신호등 인식
#     |   ├── train 
#     |   ├── val  
#     |   └── test
#     |
#     └── segmentation   <- 객체복합상태인식
#         ├── train 
#         └── test

python split_dataset_traffic_light.py --data_path ../data/detection
```

## 2. Edit dataset root dir
- /ultralytics/datasets/traffic_light.yaml
- path: ../data/detection (절대경로로 변경)


## 3. Train
Before train, Download pre-trained model . [YOLO11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)

- <주의 1>  m모델 이상부터는 batch=32일때, VRAM 48GB 이상 사용됨. (A6000에서 out of memory가 떴다.)
- <주의 2> 파라미터 name에 이름도 모델 사이즈에 맞게 잘 표기할 것.
- 훈련이 종료되면, ../output/runs/detect/traffic_yolov11_s_sgd/weights/best.pt가 만들어진다.
```
cd ultra/tests
python train_yolo11.py
```

## 4. validation
- model_filename 절대경로로 변경할 것.
```
cd ultra/tests
python val_yolo11.py
```

## 5. test
- --model_ckpt 절대경로로 변경할 것.
```
cd ultra/tests 
python test_yolo11.py --model_ckpt ../output/runs/detect/traffic_yolov11_s_sgd/weights/best.pt
```  

# Segmentation for 객체복합상태인식

## 1. Reconstructuring the train dataset to the COCO format
- data의 directory를 아래와 똑같게 만들 것.
- <주의> 이미지 형태의 Mask를 Polygon으로 변환하여 저장하게 됨. 이 과정에서 Mask의 형태가 손실될 수 있음.
- Data_Preparation.py는 segmentation_coco에 train만 생성.
```
# setting data directory

# Autonomous_AI_Challenge
# ├── ultra
# |   ├── tests
# |   └── ultralytics
# |       └── cfg/datasets 
# └── data
#     └── detection  <- 신호등 인식
#     |   ├── test
#     |   ├── train  
#     |   └── val 
#     |
#     └── segmentation   <- 객체복합상태인식 original_data_dir
#     |   ├── test
#     |   └── train 
#     └── segmentation_coco   <- new_data_dir
#         ├── images
#         |   ├── test
#         |   ├── train 
#         |   └── val
#         └── labels
#             ├── train
#             └── val

cd ultra
python Data_Preparation.py
```

### 1.3. Copy test images into the segmentation_coco directory
```
python Test_data_Preparation.py
```

### 1.2. Split the dataset into train and val sets in COCO segmentation format
```
python split_dataset_segmentation.py --data_path ../data/detection
```


## 2. train & validation
- ultralytics/cfg/datasets/Compete_segment.yaml의 path를 segmentation_coco의 절대주소로 변경할 것.
- train_yolo11_seg.py의 project를 절대주소로 변경.
(예시: /Autonomous_AI_Challenge/ultra/output/runs/segment/)
- tests directory에 [yolo11s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) 다운받기 (더 큰 모델 사용해도 됨)

```
cd ultra/tests
python train_yolo11_seg.py
```

## 3. Test (Generate result files)
```
cd ultra/tests
python Export_Result_segment.py --model_ckpt ../output/runs/segment/segmentation_yolov11_s_sgd/weights/best.pt --input_base_dir ../../data/segmentation_coco/images/test
```
