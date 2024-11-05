import os
import shutil
import random

import argparse
from tqdm import tqdm

import random

random.seed(42)


def split_dataset(directory):
    # 디렉토리 유무 확인
    if not os.path.exists(directory):
        print("data directory does not exist.")
        return

    # 훈련 디렉토리 내 labels 목록 불러오기
    train_labels_dir = os.path.join(directory, "train/labels")
    train_images_dir = os.path.join(directory, "train/images")
    files = os.listdir(train_labels_dir)

    # .txt 확장자를 가진 파일들에서 확장자 제거
    files = [os.path.splitext(file)[0] for file in files]

    # 파일 정렬
    files = sorted(files)

    # 전체 파일 개수
    total_files = len(files)

    # 10%에 해당하는 파일 수, 하나만 5%
    val_count = total_files // 10

    # 데이터를 100구간으로 나누기
    num_splits = 100
    split_size = total_files // num_splits
    splits = [files[i * split_size : (i + 1) * split_size] for i in range(num_splits)]

    # 랜덤하게 36 구간 선택
    selected_splits = random.sample(range(num_splits), 36)

    # 랜덤하게 12 구간 선택
    for i in range(3):
        selected_split = selected_splits[i * 12 : (i + 1) * 12]
        if i == 2:
            val_count = total_files // 5

        # 구간에서 전체 데이터셋의 10%를 추출해서 val 데이터에 삽입.
        val_files = []
        cnt = 0
        for split_idx in selected_split:
            split = splits[split_idx]
            cnt += len(split)
            if cnt > val_count:
                val_files.extend(split[: (len(split) - (cnt - val_count))])
                break
            else:
                val_files.extend(split)

        # 검증 데이터(val.txt)에 포함되지 않은 모든 파일을 train 데이터로 사용
        train_files = [file for file in files if file not in val_files]

        if len(train_files) + len(val_files) != total_files:
            raise ValueError(
                "The total number of training and validation data is incorrect."
            )

        # val 디렉토리 만들기.
        val_labels_path = os.path.join(directory, f"val{i}/labels")
        val_images_path = os.path.join(directory, f"val{i}/images")
        train_labels_path = os.path.join(directory, f"train{i}/labels")
        train_images_path = os.path.join(directory, f"train{i}/images")
        os.makedirs(val_labels_path, exist_ok=True)
        os.makedirs(val_images_path, exist_ok=True)
        os.makedirs(train_labels_path, exist_ok=True)
        os.makedirs(train_images_path, exist_ok=True)

        print("##### start to copy train data ######")
        for train_file in tqdm(train_files):
            source_labels_path = os.path.join(train_labels_dir, f"{train_file}.txt")
            source_images_path = os.path.join(train_images_dir, f"{train_file}.jpg")
            # 복사될 위치의 train path
            destination_train_labels_path = os.path.join(
                directory, f"train{i}/labels/{train_file}.txt"
            )
            destination_train_images_path = os.path.join(
                directory, f"train{i}/images/{train_file}.jpg"
            )

            if not os.path.exists(source_labels_path):
                raise ValueError(f"{source_labels_path} does not exist.")
            if not os.path.exists(source_images_path):
                raise ValueError(f"{source_images_path} does not exist.")

            shutil.copy(source_labels_path, destination_train_labels_path)
            shutil.copy(source_images_path, destination_train_images_path)
            break
        print("##### start to copy val data ######")
        for val_file in tqdm(val_files):
            source_labels_path = os.path.join(train_labels_dir, f"{val_file}.txt")
            source_images_path = os.path.join(train_images_dir, f"{val_file}.jpg")
            destination_val_labels_path = os.path.join(
                val_labels_path, f"{val_file}.txt"
            )
            destination_val_images_path = os.path.join(
                val_images_path, f"{val_file}.jpg"
            )

            if not os.path.exists(source_labels_path):
                raise ValueError(f"{source_labels_path} does not exist.")
            if not os.path.exists(source_images_path):
                raise ValueError(f"{source_images_path} does not exist.")

            shutil.copy(source_labels_path, destination_val_labels_path)
            shutil.copy(source_images_path, destination_val_images_path)
            break
        print(
            f"{i+1}/3th Dataset split completed. {len(val_files)} files in val and {len(train_files)} files in train"
        )
        break
    # val 중복 안 되게


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./datasets/data_traffic")
    args = parser.parse_args()
    split_dataset(args.data_path)

    print("done")


# datasets
# └── data_traffic    -> (input this direction in "--data_path")
#     ├── test
#     └── train
#         ├── images
#         |   ├── 00000000.jpg
#         |   └── 00000001.jpg ...
#         └── labels
#             ├── 00000000.txt
#             └── 00000001.txt ...
