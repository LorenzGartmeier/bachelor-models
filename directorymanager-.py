import os
import shutil
import random
import argparse

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

if __name__ == "__main__":
    main()