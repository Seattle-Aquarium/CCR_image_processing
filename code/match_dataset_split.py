"""
match_dataset_split.py
last modified: 01-Dec-2025
-----------------------------

This script copies the train/val/test split from a TEMPLATE
dataset and applies the SAME split to a TARGET dataset.

Usage:
 • TEMPLATE dataset: image patches already split into train/val/test/label folders
 • TARGET dataset:   all images currently inside train/label folders

What the script does:
 • Looks at every file in TEMPLATE/train/label, val/label, test/label
 • Finds matching filenames in TARGET/train/label
 • Moves them into TARGET/train, TARGET/val, TARGET/test so the split
   matches the template exactly

Notes:
 • DRY_RUN is currently ON (no files will be moved)
 • Set DRY_RUN = False at the top of the script to enable real moves
 • Any files in TARGET that do NOT appear in TEMPLATE will stay in train
 • Missing matches are reported
"""

import os
import shutil

SPLITS = ["train", "val", "test"]
DRY_RUN = True  # set to False to actually move files


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():

    # ---- PROMPTS FOR USER INPUT ----
    TEMPLATE_ROOT = input("Enter the path to the TEMPLATE dataset (has train/val/test): ").strip('"').strip()
    TARGET_ROOT = input("Enter the path to the TARGET dataset (all images in train): ").strip('"').strip()
    # ---------------------------------

    print(f"\nTemplate dataset: {TEMPLATE_ROOT}")
    print(f"Target dataset:   {TARGET_ROOT}")
    print(f"DRY_RUN mode:     {DRY_RUN}\n")

    moved_count = {split: 0 for split in SPLITS}
    missing_in_target = 0

    # Loop over each split in the TEMPLATE (train, val, test)
    for split in SPLITS:
        split_dir_template = os.path.join(TEMPLATE_ROOT, split)

        if not os.path.isdir(split_dir_template):
            print(f"Template split folder missing, skipping: {split_dir_template}")
            continue

        # For each label folder inside this split
        for label in os.listdir(split_dir_template):
            label_template_dir = os.path.join(split_dir_template, label)
            if not os.path.isdir(label_template_dir):
                continue

            # In TARGET: all files originally in train/label
            label_target_train_dir = os.path.join(TARGET_ROOT, "train", label)
            if not os.path.isdir(label_target_train_dir):
                print(f"[WARN] Label '{label}' not found in TARGET train; skipping.")
                continue

            # Destination dir in TARGET (train/val/test)
            label_target_dest_dir = os.path.join(TARGET_ROOT, split, label)
            ensure_dir(label_target_dest_dir)

            # Move matching files
            for fname in os.listdir(label_template_dir):
                src_path = os.path.join(label_target_train_dir, fname)
                dest_path = os.path.join(label_target_dest_dir, fname)

                if os.path.exists(src_path):
                    if DRY_RUN:
                        print(f"[DRY RUN] Would move: {src_path} -> {dest_path}")
                    else:
                        shutil.move(src_path, dest_path)
                        print(f"Moved: {src_path} -> {dest_path}")
                    moved_count[split] += 1
                else:
                    missing_in_target += 1
                    print(f"[MISSING] {fname} not found in target train for label '{label}'")

    print("\n--- Summary ---")
    for split in SPLITS:
        print(f"{split}: moved {moved_count[split]} files")

    print(f"Missing files (in template but not found in target/train): {missing_in_target} files")


if __name__ == "__main__":
    main()
