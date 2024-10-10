# How To Run Object Detection

## 1. install 
### 1.1 Install ultralytics
```
pip install ultralytics
````

### 1.2 Create conda env.
```
conda create -n ultra python=3.8 -y
conda activate ultra
cd ultralytics
pip install -e .
```

## 2. Split the dataset into training and validation sets.
```
# setting data directory

# traffic_light
# ├── ultralytics
# |   ├── tests
# |   └── ultralytics
# |       └── datasets 
# └── data
#     └── detection  <- 신호등 인식
#     |   └── train 
#     |   └── val  
#     |   └── test
#     |
#     └── segmantation   <- 객체복합상태인식
#         └── train 
#         └── val  
#         └── test

python split_dataset_traffic_light.py --data_path ../data/detection
```

## 3. Edit dataset root dir
- /ultralytics/datasets/traffic_light.yaml
- path: ../data/detection (절대경로로 변경)


## 4. Train
Before train, Download pre-trained model . [YOLO11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)

- <주의 1>  m모델 이상부터는 VRAM 48GB 이상 사용됨. (A6000에서 out of memory가 떴다.)
- <주의 2> train_yolo11.py 에서 train의 파라미터인 cache='disk' 에서 'disk'는 하드디스크에 이미지 캐싱을 저장함(70GB 이상 소요), True는 RAM으로 저장하는 것 같음.
- <주의 3> 파라미터 name에 이름도 모델 사이즈에 맞게 잘 표기할 것.
- 훈련이 종료되면, tests/runs/detect/traffic_yolov11_s_sgd/weights/best.pt가 만들어진다.
```
# cd tests
python train_yolo11.py
```

## 5. validation
- model_filename 절대경로로 변경할 것.
```
# cd tests
python val_yolo11.py
```

## 6. test
- model_filename 절대경로로 변경할 것.
```
# cd tests 
python test_yolo11.py 
```


