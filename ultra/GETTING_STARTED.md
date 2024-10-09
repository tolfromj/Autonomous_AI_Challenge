# How to run ultralytics

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
python split_dataset.py --data_path ../data
```

## 3. train
Before train, Download pre-trained model . [YOLO11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)

- <주의 1> 48GB 이어도, m모델 이상부터는 out of memory가 떴다. 
- <주의 2> train_yolo11.py 에서 train의 파라미터인 cache='disk' 에서 'disk'는 하드디스크에 이미지 캐싱을 저장하는 것 같고, True는 RAM으로 저장하는 것 같다.
- <주의 3> 파라미터 name에 이름도 모델 사이즈에 맞게 잘 표기할 것.
- 훈련이 종료되면, tests/runs/detect/traffic_yolov11_s_sgd/weights/best.pt가 만들어진다.
```
# cd tests
python train_yolo11.py
```

## 4. validation
- model_filename 절대경로로 변경할 것.
```
cd tests
python val_yolo11.py
```

## 5. test
- model_filename 절대경로로 변경할 것.
```
cd tests 
python test_yolo11.py 
```