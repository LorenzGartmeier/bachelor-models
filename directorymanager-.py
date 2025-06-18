import os
import shutil
import random
import argparse
import os
import random
import shutil


def is_image(filename):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    return filename.lower().endswith(exts)

def flatten_folder(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath == root:
            continue
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            dst = os.path.join(root, filename)
            # Avoid overwriting files with the same name
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dst):
                dst = os.path.join(root, f"{base}_{counter}{ext}")
                counter += 1
            shutil.move(src, dst)
    # Remove empty subdirectories
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        for dirname in dirnames:
            full_dir = os.path.join(dirpath, dirname)
            if os.path.isdir(full_dir):
                os.rmdir(full_dir)

def reduce_images(root, keep_count):
    images = [f for f in os.listdir(root) if is_image(f)]
    if len(images) <= keep_count:
        print(f"Already {len(images)} images or fewer. No deletion needed.")
        return
    to_delete = random.sample(images, len(images) - keep_count)
    for filename in to_delete:
        os.remove(os.path.join(root, filename))
    print(f"Deleted {len(to_delete)} images. {keep_count} remain.")

def main():

    folder = input("Enter the folder path to flatten and reduce images: ").strip()
    #keep = int(input("Enter the number of images to keep (default is 100): ").strip())
    root = os.path.abspath(folder)
    flatten_folder(root)
    #reduce_images(root, keep)

def split_images_into_train_test(data_dir, ratio):
    # Validate the ratio
    if ratio < 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1")
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.npy']
    
    # List all files in the directory
    all_files = os.listdir(data_dir)
    image_files = []
    for f in all_files:
        full_path = os.path.join(data_dir, f)
        if os.path.isfile(full_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                image_files.append(f)
    
    total_images = len(image_files)
    if total_images == 0:
        print("No images found in the directory.")
        return
    
    # Shuffle the list of images
    random.shuffle(image_files)
    
    # Calculate the number of images for training using rounding
    n_train = int(ratio * total_images + 0.5)
    n_test = total_images - n_train
    
    # Create train and test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Move images to train directory
    for i in range(n_train):
        src_path = os.path.join(data_dir, image_files[i])
        dst_path = os.path.join(train_dir, image_files[i])
        shutil.move(src_path, dst_path)
    
    # Move images to test directory
    for i in range(n_train, total_images):
        src_path = os.path.join(data_dir, image_files[i])
        dst_path = os.path.join(test_dir, image_files[i])
        shutil.move(src_path, dst_path)
    
    print(f"Split completed: {n_train} images in 'train', {n_test} images in 'test'.")

if __name__ == "__main__":
    for folder in os.listdir('baseline_model/self_descriptions/OSMA'):
        if os.path.isdir(os.path.join('baseline_model/self_descriptions/OSMA', folder)):
            dataset_path = os.path.join('baseline_model/self_descriptions/OSMA', folder)
            split_images_into_train_test(dataset_path, 0.8)