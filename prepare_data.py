"""
prepare_data.py
----------------
The original Food-11 dataset (as released by EPFL / mirrored on Kaggle)
usually comes in this layout:

    Food-11/
        training/
            0_0.jpg
            0_1.jpg
            1_0.jpg
            ...
        validation/
            0_0.jpg
            ...
        evaluation/
            0_0.jpg
            ...

Each filename is "<class_index>_<image_index>.jpg", where class_index is
0-10 matching CLASS_NAMES in utils.py.

Keras' ImageDataGenerator.flow_from_directory() (used in train.py) expects
one sub-folder per class instead:

    dataset/
        train/
            Bread/
            Dairy product/
            ...
        val/
            Bread/
            ...
        test/
            Bread/
            ...

This script reorganizes the raw Food-11 folders into that structure using
copies (originals are left untouched).

Usage:
    python prepare_data.py --source /path/to/Food-11 --dest ./dataset

If your dataset already has class-named subfolders (some Kaggle mirrors
ship it that way), you can skip this script entirely and point train.py
directly at your dataset folder.
"""

import argparse
import os
import shutil

from utils import CLASS_NAMES


def reorganize_split(src_split_dir: str, dest_split_dir: str):
    if not os.path.isdir(src_split_dir):
        print(f"  [skip] {src_split_dir} does not exist")
        return

    os.makedirs(dest_split_dir, exist_ok=True)
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(dest_split_dir, class_name), exist_ok=True)

    count = 0
    for fname in os.listdir(src_split_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            label_idx = int(fname.split("_")[0])
        except ValueError:
            print(f"  [warn] could not parse class index from '{fname}', skipping")
            continue

        if label_idx < 0 or label_idx >= len(CLASS_NAMES):
            print(f"  [warn] class index {label_idx} out of range for '{fname}', skipping")
            continue

        class_name = CLASS_NAMES[label_idx]
        src_path = os.path.join(src_split_dir, fname)
        dst_path = os.path.join(dest_split_dir, class_name, fname)
        shutil.copyfile(src_path, dst_path)
        count += 1

    print(f"  copied {count} images into {dest_split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Reorganize raw Food-11 dataset into class-named folders.")
    parser.add_argument("--source", required=True, help="Path to the raw Food-11 folder "
                                                          "(contains training/validation/evaluation subfolders)")
    parser.add_argument("--dest", default="./dataset", help="Where to write the reorganized dataset")
    args = parser.parse_args()

    mapping = {
        "training": "train",
        "validation": "val",
        "evaluation": "test",
    }

    for src_name, dest_name in mapping.items():
        src_split_dir = os.path.join(args.source, src_name)
        dest_split_dir = os.path.join(args.dest, dest_name)
        print(f"Processing '{src_name}' -> '{dest_name}' ...")
        reorganize_split(src_split_dir, dest_split_dir)

    print("\nDone. Reorganized dataset is at:", os.path.abspath(args.dest))


if __name__ == "__main__":
    main()
