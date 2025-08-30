# CCR_image_processing

## Overview
This repo is intended to aggregate information regarding the automation of our ROV survey imagery processing. 
Our overarching hope is to train a model to process our ROV survey imagery for us, such that the output files are then ready for subsequent analyses to extract data. 
Specifically, we currently use Adobe Lightroom to batch process ~150 images at a time, from a single ROV survey.
We use a de-noise feature, edit white-balance, contrast, sharpness, tone, etc., enabling an image sufficiently details for our ML approach ([CoralNet-Toolbox](https://github.com/Jordan-Pierce/CoralNet-Toolbox)) to extract percent-cover and abundance data from them. 

At present, this photo processing step is by far the most rate-limiting--it's the bottleneck preventing us from more rapidly translating ROV survey imagery --> processed and analyzed results. 
See [here](https://github.com/Seattle-Aquarium/CCR_development/blob/main/1-pagers/AI-ML_image_processing.md) for a 1-pager description of the problem on our CCR_development repo.

See below for an example of one pair of pre-processed vs hand-edited images

We have linked to imagery here to facilitate testing, model training, and workflow development. 

* [linked here](https://github.com/Seattle-Aquarium/CCR_image_processing/tree/main/example_raw_and_processed_photos) are 20 images - both the raw, original .GPR files, and the processed, hand-edited (in Adobe Lightroom) jpeg files. This smaller dataset is to enable prototyping.
* [linked here](https://www.dropbox.com/scl/fo/0jm7jocha7uj5ce9u1vvx/AILmrQqqiT18ghbCHFxDcng?rlkey=jyiiskar6n33miprjcl5qxpaz&e=1&st=v51ehwym&dl=0) are 5000 image pairs, again, both the raw .GPR and the hand-edited jpeg files. This larger dataset is to enable full-scale modeling training/deployment. 

<p float="center">
  <img src="https://github.com/user-attachments/assets/5ec9d885-c6c5-4bc1-b53b-c2b10bf78437" width="400" height="350" />
  <img src="example_raw_and_processed_photos/output_JPEG/2024_10_08_10-37-22.jpg" width="400" height="350" />
</p>


