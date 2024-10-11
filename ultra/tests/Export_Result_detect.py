from ultralytics import YOLO
import os 
import glob
import argparse

def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension.lstrip('.')

def main(model_filename):
    result_path = "../Result/detect"
    test_db_path = "../../data/detection/test"
    test_res_path = "%s/predictions"%(result_path)

    if not os.path.exists(test_res_path):
        os.makedirs(test_res_path)

    img_exts = ["jpg","bmp","png"]
    img_files = list()
    for img_ext in img_exts:
        img_files += glob.glob("%s/images/*.%s"%(test_db_path, img_ext))
                                
    img_files.sort() 

    model = YOLO(model_filename)

    # img_filename -> ../../data/detection/test/images/00001234.jpg
    for img_filename in img_files:
        result = model.predict(img_filename, imgsz=1280, conf=0.001, iou=0.6)[0]
        # result = model.predict(img_filename)[0]
        
        img_ext = get_file_extension(img_filename)
        txt_filename = img_filename.replace(img_ext, "txt")                          # ../../data/detection/test/images/00001234.txt
        txt_filename = txt_filename.replace("../data/detection/test/images","Result/detect/predictions") # ../../data/detection/test/predictions/00001234.txt
        boxes = result.boxes                                                      # => ../Result/detect/predictions
        num_obj = len(boxes.cls)

        with open(txt_filename, 'w') as f1:
            for obj_idx in range(num_obj):
                cls_id = int(boxes.cls[obj_idx])
                cs = boxes.conf[obj_idx]
                xywhn = boxes.xywhn[obj_idx] 
                # class_id norm_center_x norm_center_y norm_w norm_h confidence_score
                f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, xywhn[0], xywhn[1],xywhn[2],xywhn[3], cs))

                # xywh = boxes.xywh[obj_idx]
                # f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, cs, xywh[0], xywh[1],xywh[2],xywh[3]))


        if num_obj == 0:
            print(txt_filename)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt", type=str, default="../output/runs/detect/traffic_yolov11_s_sgd/weights/best.pt"   
    )
    args = parser.parse_args()
    main(args.model_ckpt) # 절대경로로 변경할 것.