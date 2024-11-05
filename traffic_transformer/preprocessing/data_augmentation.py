import os
import random
import shutil

import numpy as np
import albumentations as A
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# 랜덤 시드 설정
random.seed(7)

# 이미지, 레이블 디렉토리 및 타겟 이미지 설정
image_dir = "path/to/train/images" # 학습 이미지 경로
label_dir = "path/to/train/labels"
output_dir = "path/to/output" # 출력 경로 

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# 이미지 목록 생성
images = [im for im in os.listdir(image_dir) if os.path.splitext(im)[1] == '.jpg']
num_elements = 1343 # 증강할 데이터 수
random_images = sorted(random.sample(images, num_elements))

target_path = os.path.join(image_dir, '00000000.jpg')
target_image = np.array(Image.open(target_path))
target_image = np.full_like(target_image, 10)

# 변환 구성 함수
def get_transform():
    return A.Compose([
        A.FDA([target_image], p=1, read_fn=lambda x: x),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=(-10, -10), val_shift_limit=(-10, -10), p=1.0),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=-10, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ])

# 개별 이미지 변환 함수
def transform_image(img_path):
    # 이미지 열기
    random.seed(7)
    img = np.array(Image.open(img_path))
    transform = get_transform()  # 각 프로세스에서 새로운 변환 객체 생성
    aug = transform(image=img)['image']
    return aug, img_path

# 변환된 이미지 및 labels 저장 함수
def save_transformed_image(aug, img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, 'images', f"5{name[1:]}.jpg")
    source_labels_path = os.path.join(label_dir, f'{name}.txt')
    destination_labels_path = os.path.join(output_dir, 'labels', f'5{name[1:]}.txt')

    shutil.copy(source_labels_path, destination_labels_path)
    transformed_img = Image.fromarray(aug)
    transformed_img.save(output_path)

# 병렬 처리
with ProcessPoolExecutor() as executor:
    # 이미지 경로 생성
    img_paths = [os.path.join(image_dir, image) for image in random_images]

    for img_path in tqdm(img_paths):
        aug, img_path = transform_image(img_path)  # 변환 수행
        save_transformed_image(aug, img_path)