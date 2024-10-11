from ultralytics import YOLO


model = YOLO("yolo11s-seg.yaml")
# Train the model with 2 GPUs
results = model.train(data="Compete_segment.yaml",
                      project='/workspace/traffic_light/ultra/output/runs/segment/', # 절대경로로 바꿔주기. 
                      name="segmentation_yolov11_s_sgd",
                      pretrained='yolo11s-seg.pt',
                      batch=140,
                      imgsz=32,
                      epochs=1,
                      cache=False, 
                      device= 0) # [0,1,2,3,4,5,6,7])
