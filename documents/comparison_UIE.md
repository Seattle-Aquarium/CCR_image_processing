# SOP for Comparing UIE Output to Hand-Edited Images for Classification Training

This SOP describes how to prepare matched datasets (hand-edited vs. UIE), train Ultralytics YOLO classification models in the [CoralNet Toolbox](https://github.com/Jordan-Pierce/CoralNet-Toolbox) (v0.0.97), and compare model performance to evaluate how image enhancement affects classifier accuracy.

---

## 1. Toolbox Installation and Setup

Instructions for installing and running Toolbox are available here:  
<https://www.dropbox.com/scl/fi/ut3k3invqvyhyyj168332/Toolbox_installation.docx?rlkey=qob1tuoqmnaljc87blidohpx3&dl=0>

---

## 2. Download Hand-Edited Images

- Download hand-edited images:  
  <https://www.dropbox.com/scl/fo/j2or8e3jh99spm1hd84ws/AMI4BBG_Jfai-AsElzKYjxw?rlkey=kfy6gp1up0go9x1laoyo6lw83&dl=0>
- Import images into Toolbox:  
  **File → Import → Rasters → Images**

---

## 3. Import Annotations for Hand-Edited Images

- Import JSON annotation file:  
  **File → Import → Annotations → JSON**
- Annotation file:  
  <https://www.dropbox.com/scl/fi/n00ye2ybpo9csywqznh18/hand_edited_annotations.json?rlkey=kozunfiw9ck9986e0m1jxo348&dl=0>

---

## 4. Export the Hand-Edited Dataset

Export annotations into a classification dataset:

- **File → Export → Dataset → Classify**
- Train/val/test split: **0.7 / 0.2 / 0.1**

This dataset serves as the *template* for splitting the UIE dataset.

---

## 5. Prepare UIE Images

Ensure the UIE images correspond **exactly** to the hand-edited images:

- Same filenames  
- Same image dimensions  
- Same number of images  

---

## 6. Import UIE Images into Toolbox

- **File → Import → Rasters → Images**

---

## 7. Edit Hand-Edited JSON Annotation File to Point to UIE Images

Because UIE images are stored in a different folder, the annotation JSON from the hand-edited set must be updated so that every `"image_path"` points to the UIE directory.

### Run the JSON path update script

Use `update_annotation_paths.py` to automatically update all `"image_path"` entries.

The script will:

- Ask for the path to the *original* annotation JSON  
- Ask for the path to the **UIE image folder**  
- Ask where the updated JSON should be saved (and what to name it)  
- Replace only the directory portion of each `"image_path"` (filenames remain unchanged)  
- Output a new JSON annotation file ready for import into Toolbox

---

## 8. Import the UIE Annotations

- Import the updated JSON file produced in Step 7:
  - **File → Import → Annotations → JSON**

---

## 9. Export the UIE Dataset

Export the UIE annotations to a dataset:

- **File → Export → Dataset → Classify**
- Train/val/test split: **1.0 / 0.0 / 0.0**

This creates a dataset that will later be re-split to match the hand-edited dataset.

---

## 10. Match the UIE Dataset to the Hand-Edited Split

Use `match_dataset_split.py` to apply the same train/val/test split as the hand-edited dataset.

The script:

- Reads all files in **TEMPLATE/train/labels**, **val/labels**, **test/labels**
- Finds matching filenames in the UIE dataset  
- Moves UIE label files into **TARGET/train**, **TARGET/val**, and **TARGET/test**  
- Ensures *identical image patches* are used in both datasets

**Outcome:**  
Two datasets containing the *same* image patches, enabling a fair and controlled comparison of classification model performance.

---

## 11. Train Classification Models  
*(Train once for the hand-edited dataset and once for the UIE dataset)*

- Ultralytics → **Train Model → Classify**
- In the training window:
  - **Dataset:** Browse → select dataset root folder  
  - **Model:** `YOLO11s-cls`  
  - **Parameters:**  
    - Set save location for trained model  
    - Use default training hyperparameters
- Click **OK** to begin training  
Training progress appears in the terminal.

---

## 12. Evaluate Training Results

After training completes, examine the model output folder and review:

### `results.csv`

- **Best validation Top-1 accuracy**  
  → Primary measure of classifier performance

- **Best validation Top-5 accuracy**  
  → Higher Top-5 suggests richer or more informative images

- **Validation loss**  
  → Lower loss at similar accuracy indicates better calibration

### `metrics_report.csv` (found in the test subfolder)

- **Macro F1**  
  → Mean performance across all classes (equal weighting)

- **Macro Balanced Accuracy**  
  → Mean balanced accuracy across classes

- **Weighted F1**  
  → F1 score weighted by class sample size

---

### Optional: Automated Comparison

Run `ML_metrics_comparison.py` to automatically generate a comparison table showing differences in accuracy, F1 scores, and other metrics between the hand-edited and UIE models.

---
