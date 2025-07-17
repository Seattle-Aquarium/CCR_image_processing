# CCR_image_processing

## Overview
This repo is intended to aggregate information regarding the automation of our ROV survey imagery processing. 
Our overarching hope is to train some sort of ML framework to process our ROV survey imagery for us, such that the output files are then ready for subsequent analyses to extract data. 
Specifically, we currently use Adobe Lightroom to batch process ~150 images at a time, from a single ROV survey.
We use a de-noise feature, edit white-balance, contrast, sharpness, tone, etc., enabling an image sufficiently details for our ML approach ([CoralNet-Toolbox](https://github.com/Jordan-Pierce/CoralNet-Toolbox)) to extract percent-cover and abundance data from them. 

At present, this photo processing step is by far the most rate-limiting--it's the bottleneck preventing us from more rapidly translating ROV survey imagery --> processed and analyzed results. 

With ~ 5000 fully processed images (with both the "raw" original image and the "processed" output image in hand), we hope to, e.g., train a Generative Adversarial Network (GAN) to process our imagery for us. 
We have ~ 7000 images awaiting processing, with more coming this upcoming field season (August, September 2025). 

See [here](https://github.com/Seattle-Aquarium/CCR_development/blob/main/1-pagers/AI-ML_image_processing.md) for a 1-pager description of the problem on our CCR_development repo. 

To provide example imagery, a folder containing 10 raw (pre-processing, .GPR), preview (pre-processing, .JPEG), and polished (processed, JPEG) can be found [here](https://github.com/zhrandell/CCR_image_processing/tree/main/example_raw_and_processed_photos). 

<p float="center">
  <img src="example_raw_and_processed_photos/input_JPEG/2024_10_08_10-37-22.JPG" width="400" height="350" />
  <img src="example_raw_and_processed_photos/output_JPEG/2024_10_08_10-37-22.jpg" width="400" height="350" />
 </p>


## General information; workflows ready to implement
The following repos contain general information about our work, and specialized repos for ROV telemetry analyses, processing and analyses of ROV-derived benthic abundance and distribution data, and simulating benthic data.  

```mermaid
graph TD

A["<a href='https://github.com/Seattle-Aquarium/Coastal_Climate_Resilience' target='_blank' style='font-size: 16px; font-weight: bold;'>Coastal_Climate_Resilience</a><br><font color='darkgray'>the main landing pad for the CCR research program</font>"]

A --> E["<a href='https://github.com/Seattle-Aquarium/CCR_analytical_resources' target='_blank' style='font-size: 16px; font-weight: bold;'>CCR_ROV_telemetry_processing</a><br><font color='darkgray'>analytical tools for working with ROV telemetry data</font>"]

A --> F["<a href='https://github.com/Seattle-Aquarium/CCR_benthic_analyses' target='_blank' style='font-size: 16px; font-weight: bold;'>CCR_benthic_analyses</a><br><font color='darkgray'>code to work with ROV-derived benthic community data</font>"]

A --> G["<a href='https://github.com/Seattle-Aquarium/CCR_benthic_taxa_simulation' target='_blank' style='font-size: 16px; font-weight: bold;'>CCR_benthic_taxa_simulation</a><br><font color='darkgray'>code to simulate ROV-derived benthic community</font>"]
```

## Help wanted! 
The following repos involve active areas of open-source software development, AI/ML implementation, and computer vision challenges; areas where we could use assistance are 🔶 highlighted in orange 🔶

```mermaid
graph TD

B["<a href='https://github.com/Seattle-Aquarium/CCR_development' target='_blank' style='font-size: 16px; font-weight: bold;'>CCR_development</a><br><font color='darkgray'>main hub for organizing active Issues under development </font>"]

B --> C["<a href='https://github.com/Seattle-Aquarium/CCR_image_processing' target='_blank' style='font-size: 16px; font-weight: bold;'>CCR_image_processing</a><br><font color='darkgray'>help wanted to implement AI/ML solution to expendite image processing</font>"]

B --> D["<a href='https://github.com/Seattle-Aquarium/CCR_kelp_feature_detection' target='_blank' style='font-size: 16px; font-weight: bold;'>CCR_kelp_feature_detection</a><br><font color='darkgray'>active research re: photogrammetry in kelp forests</font>"]

style B stroke:#FF8600,stroke-width:4px
style C stroke:#FF8600,stroke-width:4px
```
