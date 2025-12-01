

```bibtex
@misc{williams2025,
  author = {Williams, Megan},
  title = {SOP for Comparing UIE Output to Hand-Edited Images},
  institution = {Seattle Aquarium},
  date = {2025-12-01}
}
```

# SOP for Comparing UIE Output to Hand-Edited Images

The following steps are required to create a training dataset, train a classification model using Ultralytics YOLO in [Toolbox](https://github.com/Jordan-Pierce/CoralNet-Toolbox) (v0.0.97), and apply the model to make predictions.

## 1. Toolbox installation and setup

- Instructions for installing and running Toolbox can be found in the documentation linked [here](https://www.dropbox.com/scl/fi/ut3k3invqvyhyyj168332/Toolbox_installation.docx?rlkey=qob1tuoqmnaljc87blidohpx3&dl=0).

## 2. Download hand-edited images 

- Download hand-edited images linked [here](https://www.dropbox.com/scl/fo/j2or8e3jh99spm1hd84ws/AMI4BBG_Jfai-AsElzKYjxw?rlkey=kfy6gp1up0go9x1laoyo6lw83&dl=0).
- Import images to Toolbox 
  - File → Import → Rasters → Images

## 3. Import annotations 

- Import JSON annotation file 
  - File → Import → Annotations → JSON
  - Our JSON annotations for the hand-edited imagery can be download [here][(https://www.dropbox.com/scl/fi/7rbkh40zzj7xbjx4nydoc/labelset_31.json?rlkey=7curccvmqin4ia1xqazum4h3m&dl=0)]. 

## 4. Export dataset

- Export hand-edited annotations into a dataset 
  - File → Export → Dataset → Classify
  - Train/val/test ratios: 0.7/0.2/0.1 


## 5. Download UIE edited images

- The UIE images need to be the *same* images from the hand-edited set, containing the same dimensions and image name. 

## 6. Upload UIE images to Toolbox

- Import images to Toolbox 
  - File → Import → Rasters → Images

## 7. Import annotations from hand-edited images (same as Step 3)

- Import JSON annotation file 
  - File → Import → Annotations → JSON
  - Our JSON annotations for the hand-edited imagery can be download [here][(https://www.dropbox.com/scl/fi/7rbkh40zzj7xbjx4nydoc/labelset_31.json?rlkey=7curccvmqin4ia1xqazum4h3m&dl=0)]. 


## 8. Export UIE dataset

- Export UIE annotations into the train folder of a dataset
  - File → Export → Dataset → Classify
  - Train/val/test ratios: 1.0/0.0/0.0 

## 9. Match UIE dataset to hand-edited dataset

- Match the UIE dataset to the hand-edited dataset 
  - Use match_dataset_split.py 
  - This scripts tooks at every file in TEMPLATE/train/label, val/label, test/label. 
  - Finds matching filenames in TARGET/train/label
  - Moves them into TARGET/train, TARGET/val, TARGET/test so the split
   matches the template exactly
- The result is two datasets that have the same image patches to be used to train a classification model. 

## 10. Train classification models (repeat for hand-edited and UIE dataset)

- Start training a YOLO classification model
  - Ultralytics → Train Model → Classify
- In the training window:
  - Dataset: Click Browse and select the exported dataset folder.
  - Model Selection: YOLO11s-cls
  - Parameters:
    - Set the location where you want your trained model to be saved.
    - Use default training parameters. 
- Click OK to begin training. You can monitor training progress in the terminal.

## 11. Evaluate training results 
- Training results can be seen in the terminal and upon the completion of the training in the location designated in Step 10. 
- 