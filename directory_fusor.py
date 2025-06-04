import os
import shutil

def fuse_directories(src_dir, dst_dir):
    """
    Merge contents of src_dir into dst_dir. Overwrites files with the same name.
    """
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_root, file)
            shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    fuse_directories('datasets/imagenet/val', 'datasets/imagenet/train')