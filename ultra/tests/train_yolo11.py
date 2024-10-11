from ultralytics import YOLO
import os 
os.environ["NCCL_P2P_DISABLE"] = "1" 



# Load a model
model = YOLO('yolov11s.pt')  # load a pretrained model (recommended for training)

#########################################################
# Train the model
model.train(data='../ultralytics/cfg/datasets/traffic_light.yaml', 
            project='/workspace/traffic_light/ultra/output/runs/detect/',
            name="traffic_yolov11_s_sgd",
            epochs=30, 
            imgsz=1280,  
            device="0", # "0,1,2,3,4,5,6,7"
            batch=32,
            cache='disk', # True로 하면 이미지를 모두 RAM에 올려서 46GB를 차지하게 됨. 조심!
            pretrained=True, 
            lr0 = 0.001,
            fliplr = 0.0, 
            mosaic = 1.0,
            optimizer='SGD',
            close_mosaic=5)