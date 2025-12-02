# SOP for Comparing UIE Output to Hand-Edited Images for Classification Training

This SOP describes how to prepare matched datasets (hand-edited vs. UIE), train Ultralytics YOLO classification models in [Toolbox](https://github.com/Jordan-Pierce/CoralNet-Toolbox) (v0.0.97), and compare model performance to evaluate image quality.



## 1. Toolbox installation and setup

- Instructions for installing and running Toolbox can be found in the documentation linked [here](https://www.dropbox.com/scl/fi/ut3k3invqvyhyyj168332/Toolbox_installation.docx?rlkey=qob1tuoqmnaljc87blidohpx3&dl=0).

## 2. Download hand-edited images 

- Download hand-edited images linked [here](https://www.dropbox.com/scl/fo/j2or8e3jh99spm1hd84ws/AMI4BBG_Jfai-AsElzKYjxw?rlkey=kfy6gp1up0go9x1laoyo6lw83&dl=0).
- Import images to Toolbox 
  - File → Import → Rasters → Images

## 3. Import annotations for hand-edited images

- Import JSON annotation file 
  - File → Import → Annotations → JSON
  - Our JSON annotations for the hand-edited imagery can be download [here](https://www.dropbox.com/scl/fi/7rbkh40zzj7xbjx4nydoc/labelset_31.json?rlkey=7curccvmqin4ia1xqazum4h3m&dl=0). 

## 4. Export hand-edited dataset

- Export hand-edited annotations into a dataset 
  - File → Export → Dataset → Classify
  - Train/val/test split: 0.7 / 0.2 / 0.1 


## 5. Prepare UIE edited images

- Ensure UIE outputs correspond **exactly** to the hand-edited images:
  - Same filenames  
  - Same image dimensions  
  - Same number of images  

## 6. Import UIE images into Toolbox

- File → Import → Rasters → Images

## 7.  Import the Same Annotations (Used for Hand-Edited Images)

- Import JSON annotation file 
  - File → Import → Annotations → JSON
- Use the same annotation file as in Step 3: linked [here](https://www.dropbox.com/scl/fi/7rbkh40zzj7xbjx4nydoc/labelset_31.json?rlkey=7curccvmqin4ia1xqazum4h3m&dl=0). 


## 8. Export UIE dataset

- Export UIE annotations into the train folder of a dataset
  - File → Export → Dataset → Classify
  - Train/val/test split: 1.0 / 0.0 / 0.0 
- This creates a dataset that will be re-split in Step 9 to match the hand-edited dataset in the next step.

## 9. Match UIE dataset to hand-edited dataset

Use `match_dataset_split.py` to apply the same train/val/test split used for the hand-edited dataset.

- The script: 
  - Reads all files in **TEMPLATE/train/labels**, **val/labels**, **test/labels**
  - Finds matching filenames in **TARGET/train/labels**
  - Moves UIE files into **TARGET/train**, **TARGET/val**, **TARGET/test** so the split matches exactly
  - Moves them into TARGET/train, TARGET/val, TARGET/test so the split
   matches the template exactly

Outcome:
- Two datasets containing the *same* image patches, enabling a fair comparison of model performance.


## 10. Train classification models (repeat for hand-edited and UIE dataset)
(Repeat this process once for the hand-edited dataset and once for the UIE dataset)

- Start training a YOLO classification model
  - Ultralytics → Train Model → Classify
- In the training window:
  - Dataset: Click Browse and select a dataset folder.
  - Model Selection: YOLO11s-cls
  - Parameters:
    - Set the location where you want your trained model to be saved.
    - Use default training parameters. 
- Click OK to begin training. You can monitor training progress in the terminal.

## 11. Evaluate training results 

After training completes go to the folder created for the trained model and review metrics from:

`results.csv`
- **Best validation Top-1 accuracy**  
  - Does this dataset improve classification accuracy?
- **Best validation Top-5 accuracy**  
  - If Top-1 is similar but Top-5 is higher, the dataset provides “richer” information.
- **Validation loss**  
  - Lower loss at similar accuracy indicates better model calibration.

`metrics_report.csv` in the test folder within the model folder
- **Macro F1**  
  - Performance across classes, treating each class equally.
- **Macro Balanced Accuracy**  
  - Mean of per-class balanced accuracy.
- **Weighted F1**  
  - Class-wise F1 weighted by sample counts.


### Optional: Automated Comparison

Run `ML_metrics_comparison.py` to automatically generate a comparison table of metrics for the hand-edited vs. UIE models.
