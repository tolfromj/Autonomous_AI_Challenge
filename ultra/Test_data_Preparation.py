import os
import shutil
import argparse

def main(source_dir, destination_dir):
    # 대상 경로가 존재하지 않으면 생성
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 서브 폴더를 순회하며 이미지 파일 복사 및 이름 변경
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder, 'img')
        
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            # 서브 폴더 내 이미지 파일을 순회
            for image_file in os.listdir(subfolder_path):
                if image_file.endswith('.png'):
                    # 원본 이미지 파일 경로
                    source_image_path = os.path.join(subfolder_path, image_file)
                    
                    # 새로운 파일 이름 생성
                    new_image_name = f"{subfolder}_{image_file}"
                    
                    # 대상 이미지 파일 경로
                    destination_image_path = os.path.join(destination_dir, new_image_name)
                    
                    # 파일 복사
                    shutil.copy(source_image_path, destination_image_path)
                    print(f"Copied {source_image_path} to {destination_image_path}")

    print("All files have been copied successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", type=str, default="../data/segmentation/test"   
    )
    parser.add_argument(
        "--destination_dir", type=str, default="../data/segmentation_coco/images/test"   
    )
    args = parser.parse_args()
    main(args.source_dir, args.destination_dir)