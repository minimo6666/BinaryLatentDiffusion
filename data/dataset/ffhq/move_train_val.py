import os
import shutil
from pathlib import Path

def copy_images_based_on_filelist():
    # 定义目录路径
    source_dir = "/mnt/data/0/mohao/data/ffhq/ffhq256"
    target_dir = "/mnt/data/0/mohao/data/ffhq/ffhq256_val_train"
    dataset_dir = "/home/minimo/Project/BinaryLatentDiffusion/data/dataset/ffhq"
    
    # 创建目标目录（如果不存在）
    train_target_dir = os.path.join(target_dir, "train")
    valid_target_dir = os.path.join(target_dir, "valid")
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(valid_target_dir, exist_ok=True)
    
    # 定义文件列表路径
    train_list_path = os.path.join(dataset_dir, "ffhqtrain.txt")
    valid_list_path = os.path.join(dataset_dir, "ffhqvalidation.txt")
    
    # 读取训练集文件列表
    with open(train_list_path, 'r') as f:
        train_files = [line.strip() for line in f.readlines() if line.strip()]
    
    # 读取验证集文件列表
    with open(valid_list_path, 'r') as f:
        valid_files = [line.strip() for line in f.readlines() if line.strip()]
    
    # 复制训练集图片
    copied_count = 0
    for filename in train_files:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(train_target_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"警告: 源文件不存在 {src_path}")
    
    print(f"成功复制 {copied_count} 个训练图片到 {train_target_dir}")
    
    # 复制验证集图片
    copied_count = 0
    for filename in valid_files:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(valid_target_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"警告: 源文件不存在 {src_path}")
    
    print(f"成功复制 {copied_count} 个验证图片到 {valid_target_dir}")

if __name__ == "__main__":
    copy_images_based_on_filelist()

source_dir = "/mnt/data/0/mohao/data/ffhq/ffhq256"

target_dir = "/mnt/data/0/mohao/data/ffhq/ffhq256_val_train"

