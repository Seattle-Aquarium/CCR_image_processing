import json
import os
import sys

def to_json_slashes(path):
    """Convert Windows path to JSON-safe forward slashes."""
    return path.replace("\\", "/")

# --- Prompt user for input paths ---
annotation_path = input("Enter the path to the original annotation JSON file: ").strip()
uie_folder = input("Enter the path to the UIE image folder: ").strip()
output_path = input("Enter the full output path and filename for the updated JSON (e.g., C:/path/annotation_UIE.json): ").strip()

# Normalize & convert to forward slashes
annotation_path = to_json_slashes(os.path.abspath(annotation_path))
uie_folder = to_json_slashes(os.path.abspath(uie_folder))
output_path = to_json_slashes(os.path.abspath(output_path))

# --- Validate annotation file ---
if not os.path.isfile(annotation_path):
    print(f"\nERROR: Annotation file not found:\n{annotation_path}")
    sys.exit(1)

# --- Validate UIE folder ---
if not os.path.isdir(uie_folder):
    print(f"\nERROR: UIE image folder not found:\n{uie_folder}")
    sys.exit(1)

# --- Validate output directory ---
output_dir = os.path.dirname(output_path)
if not os.path.isdir(output_dir):
    print(f"\nERROR: Output directory does not exist:\n{output_dir}")
    sys.exit(1)

# --- Load JSON ---
try:
    with open(annotation_path, "r") as f:
        data = json.load(f)
except json.JSONDecodeError:
    print("\nERROR: The annotation file is not valid JSON.")
    sys.exit(1)

if not isinstance(data, dict):
    print("\nERROR: Expected the JSON to be a dictionary mapping image paths to annotation lists.")
    sys.exit(1)

updated_count = 0
missing_images = []
new_data = {}

# --- Update image paths in keys and annotations ---
for old_image_path, ann_list in data.items():
    filename = os.path.basename(old_image_path)

    # Build new path with forward slashes
    new_image_path = to_json_slashes(os.path.join(uie_folder, filename))

    # Check image existence
    if not os.path.isfile(new_image_path):
        missing_images.append(filename)

    # Update image_path inside annotations
    for ann in ann_list:
        if isinstance(ann, dict):
            ann["image_path"] = new_image_path
            updated_count += 1

    # Store under new key with JSON-style slashes
    new_data[new_image_path] = ann_list

# --- Write updated JSON ---
with open(output_path, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"\nUpdated {updated_count} annotation entries.")
print(f"Updated annotation file saved to:\n{output_path}")

if missing_images:
    print("\nWARNING: The following files were NOT found in the UIE folder:")
    for fname in sorted(set(missing_images)):
        print("  -", fname)
else:
    print("\nAll annotation image filenames matched successfully.")
