# Wildfire-Detection-ChapterX
This repository presents an in-depth exploration of convolutional neural networks (CNNs), specifically U-Nets for wildfire detection. Leveraging Landsat-8 satellite imagery, the project evaluates the adaptability and performance of CNNs through various architectural configurations. It covers the entire workflow, from data collection and preprocessing to model training, evaluation, and inference generation. This repository offers hands-on experience on how UNet CNNs can enhance detection accuracy through multispectral imaging.

# Table of Contents
* [Project Background](#Wildfire-Detection-ChapterX)
* [Experimential Learning Objective & Activities](#objectives)
* [Project Set Up](#project-set-up)
* [Data](#data)
  - [A. Data Download](#A-Download-Data-from)
    - [i. Accessing Data](#i-accessing-data)
  - [B. Data Preprocessing](#B-Data-Preprocessing)
    - i. Cleaning
    - ii. Data Split
* Modeling 

# Objectives

### Prerequisites: 
You need a package manager installed to install the required packages, set up a virtual enviroment, and launch Jupyter Notebook. 
For this project we will be using Anaconda. (for installation and help: )
1. Create a conda enviroment
   ``` conda create -n <env-name> ```
2. Activate enviroment
   ``` conda activate <env-name> ```
3. Required packages list
   - Tensorflow (we will  more import packages from this library)
   - Numpy, Pandas, 
   - Os, Sys, CV2, csv, warnings, rasterio
   - threading, random, glob, time
   - Scikit (we will  more import packages from this library)
   - WandB
   
## Project Set Up
This part of our guide will walk you through setting up the directory structure for the wildfire detection experiments. This Project set up will include installing required packages and preparing the enviroment for execution. 

- First create a project directory (WildfireDetection), on you local machine:
```
mkdir WildfireDetection
cd WildfireDetection
```
- Inside the project directory, create a data folder to store datasets and other required project files
```
mk Data
``` 

## Data 
This part of our guide assumes you have a empty `Data` directory. Here we will download the data required for this study.   

### A. Download Data from:
<Link to onedrive>
Uzip folder to `Data` directory
After Unziping folder the extracted image patches and mask into the local project directory for Data. These images will have to be in two directories. One for the image patches and on for the corresponding masks these will be stored for training and evaluating the Unet CNN model. 

##### i. Accessing Data 
```
# get original Patches & Masks
images_dir = '/Users/local/WildfireDetection/Data/images/patches'
masks_dir = '/Users/local/WildfireDetection/Data/images/masks/patches'

image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)

print(f"Number of images: {len(image_files)}")
print(f"Number of masks: {len(mask_files)}")
```
### B. Data Preprocessing
At this point in the guide we assume you have two working directories for the images pateches and mask. In the next steps we will organize the dataset based on the research group (e.i.: Kumar-Roy, Murphy, Schroeder). 

