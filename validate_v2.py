import os
import random
import shutil
from tqdm import tqdm

def sample_images(src_dir, dest_dir, num_samples=100):
    """
    從源資料夾的每個標籤資料夾中隨機抽取指定數量的圖片，並將其存儲到目標資料夾。
    
    :param src_dir: 源資料夾路徑，包含標籤資料夾
    :param dest_dir: 目標資料夾路徑
    :param num_samples: 每個標籤隨機抽取的圖片數量
    """
    # 確保目標資料夾存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
    
    # 遍歷標籤資料夾
    for label in tqdm(os.listdir(src_dir), desc="Processing labels"):
        label_src_path = os.path.join(src_dir, label)
        label_dest_path = os.path.join(dest_dir, label.lower())
        
        # 如果是資料夾，則開始處理
        if os.path.isdir(label_src_path):
            # 確保目標標籤資料夾存在
            if not os.path.exists(label_dest_path):
                os.makedirs(label_dest_path)
                print(f"Created directory: {label_dest_path}")
            
            # 隨機抽取圖片
            images = os.listdir(label_src_path)
            if not images:
                print(f"No images found in: {label_src_path}")
                continue
            
            sampled_images = random.sample(images, min(num_samples, len(images)))
            
            # 複製圖片到新資料夾
            for img in sampled_images:
                src_img_path = os.path.join(label_src_path, img)
                dest_img_path = os.path.join(label_dest_path, img)
                shutil.copy(src_img_path, dest_img_path)
                print(f"Copied: {src_img_path} -> {dest_img_path}")

# 指定來源和目標資料夾路徑
src_directory = "E:/Sign-Language-B_CNN/dataset/ASL_Alphabet_Dataset/asl_alphabet_train"  # 替換為您的來源資料夾路徑
dest_directory = "E:/Sign-Language-B_CNN/dataset/small_dataset_2"  # 替換為您的目標資料夾路徑

sample_images(src_directory, dest_directory, num_samples=100)
