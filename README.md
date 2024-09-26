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
    - [i. Cleaning](#i-Cleaning)
    - [ii. Data Split](#ii-Split-Data)
    - [iii. Map Data](#iii-Map-Data)
* [Modeling](#Modeling)
  - [A. Parameter Selection](#A-Parameter-Selection)
  - [B. Data Generators](#B-Data-Generators)
  - [C. Model Archeticture](#C-Model-Archeticture)

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
   - Os, Sys, CV2, csv, warnings, rasterio,
   - threading, random, glob, time,
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
mkdir Data
``` 

You will then download the data and launch jupyter notebook before running the script to get original Patches & Masks.
``` Jupyter Notebook``` 

## Data 
This part of our guide assumes you have a empty `Data` directory. Here we will download the data required for this study.   

### A. Download Data from:
<Link to onedrive>
Uzip folder to `Data` directory
After Unziping folder the extracted image patches and mask into the local project directory for Data. These images will have to be in two directories. One for the image patches and on for the corresponding masks these will be stored for training and evaluating the Unet CNN model. 

#### i. Accessing Data 
```
# get original Patches & Masks
images_dir = '/Users/local/WildfireDetection/Data/images/patches'
masks_dir = '/Users/local/WildfireDetection/Data/images/masks/patches'

image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)

print(f"Number of images: {len(image_files)}")
print(f"Number of masks: {len(mask_files)}")
```
NOTE: the Number of images: 3031 & the Number of masks: 4385

### B. Data Preprocessing
At this point in the guide we assume you have two working directories for the images pateches and mask. In the next steps we will organize the dataset based on the research group (e.i.: Kumar-Roy, Murphy, Schroeder). Each study will go to a respective organized directory that will willl generate in the next steps.
```
### ID dataset by research group 
# Read in all files in the directory
mask_files = os.listdir(masks_dir)

# Dictionary to count processing types
type_counts = {"Kumar-Roy": 0, "Murphy": 0, "Schroeder": 0}

# Iterate through the files and count each type
for file in mask_files:
    if file.endswith(".tif"):  
        if "Kumar-Roy" in file:
            type_counts["Kumar-Roy"] += 1
        elif "Murphy" in file:
            type_counts["Murphy"] += 1
        elif "Schroeder" in file:
            type_counts["Schroeder"] += 1

# Print the counts for each type
for type_name, count in type_counts.items():
    print(f"Number of {type_name} files: {count}")
```
NOTE: the Number of Kumar-Roy files: 2619 & 
      the Number of Murphy files: 1156 & 
      the Number of Schroeder files: 610

#### i. Cleaning 
Next we will match images with their corresponding masks based on the file names and save these pairs to a CSV file. This CSV will be used for training, validation, and testing splits. 

EDIT: mask_base = os.path.splitext(mask_file)[0].replace("_Kumar-Roy", "")
      change ...("_<Study>", "")
      
In this script we are interested in the Kumar Roy Study 
```
# Create new pairs to CSV file based on Kumar-Roy 
csv_output_file = 'image_mask_pairs.csv'  

# Read in all files in the directory
image_files = os.listdir(images_dir)

# Store image and mask file pairs
matching_pairs = []

# Iterate through mask files and find corresponding images
for mask_file in mask_files:
    if mask_file.endswith(".tif"):  
        
        ### Extract base filename without extension and "TYPE" ###
        mask_base = os.path.splitext(mask_file)[0].replace("_Kumar-Roy", "")
        
        # Search for corresponding image file in images_dir
        found_match = False
        for image_file in os.listdir(images_dir):
            if image_file.endswith(".tif"):
                image_base = os.path.splitext(image_file)[0]
                if mask_base == image_base:
                    matching_pairs.append((image_file, mask_file))
                    found_match = True
                    break

# Print the number of matching pairs found
print(f"Number of corresponding image-mask pairs found: {len(matching_pairs)}")

# Write pairs to CSV file
with open(csv_output_file, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image File', 'Mask File'])
    for pair in matching_pairs:
        csv_writer.writerow(pair)

print(f"CSV file saved with {len(matching_pairs)} pairs: {csv_output_file}")
```
NOTE: Number of corresponding image-mask pairs found: 2619
      CSV file saved with 2619 pairs: image_mask_pairs.csv

#### ii. Split Data 
Next we want to split our data for Training, Validation, and Testing subsets. To ensure that the model is trained and validated properly. We will be spliting the data 60% for training, 30% for testing and 10% for validation. 
```
CSV_FILE = 'image_mask_pairs.csv'
TEST_SIZE = 0.3    # 70/ 30 split training/ testing 
VAL_SPLIT = 0.50   # 10/ 10 split testing/ validation
RANDOM_STATE = 42

IMAGES_PATH = '/Users/local/WildfireDetection/Data/organized/images'
MASKS_PATH = '/Users/local/WildfireDetection/Data/organized/masks'

# Read in all files in the directory
df = pd.read_csv(CSV_FILE)

# Shuffle the DataFrame if required
# df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Split the data
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
val_df, test_df = train_test_split(test_df, test_size=VAL_SPLIT, random_state=RANDOM_STATE)

# Extract file paths into lists
train_images = train_df['Image File'].tolist()
train_masks = train_df['Mask File'].tolist()

val_images = val_df['Image File'].tolist()
val_masks = val_df['Mask File'].tolist()

test_images = test_df['Image File'].tolist()
test_masks = test_df['Mask File'].tolist()

# Output the number of samples in each split
print(f"Training set: {len(train_images)} pairs")
print(f"Validation set: {len(val_images)} pairs")
print(f"Test set: {len(test_images)} pairs")
```

NOTE for Kumar: Training set: 1833 pairs
                Validation set: 393 pairs
                Test set: 393 pairs
#### iii. Map Data              
Finally, we will map the file names to their full paths, which will be used to load the data during model training and evaluation. 
```
"""
    X: Images 
    Y: Mask
"""
# Map the images and mask paths
images_train = [os.path.join(IMAGES_PATH, image) for image in train_images]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in train_masks]

images_validation = [os.path.join(IMAGES_PATH, image) for image in val_images]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in val_masks]

images_test = [os.path.join(IMAGES_PATH, image) for image in test_images]
masks_test = [os.path.join(MASKS_PATH, mask) for mask in test_masks]
```

# Modeling
Our model was developed for wildfire detection over the Canada Region utilizing a U-Net architecture, a popular choice for image segmentation task. 

### A. Parameter Selection
We will set an output directory to store our model results, initally we evaluate the training and validation accuarcy and loss. We will normilize our 256x256 pixel image pairs. Finally define the hyperparameters used to train our model. 
```
OUTPUT_DIR = './train_output/'

MAX_PIXEL_VALUE = 65535 # Max. pixel value, used to normalize the image

EPOCHS = 20
BATCH_SIZE = 64
IMAGE_SIZE = (256, 256)
N_CHANNELS = 3
N_FILTERS = 16

MASK_ALGORITHM = 'Kumar-Roy'

WORKERS = 4
EARLY_STOP_PATIENCE = 5 
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format('CNNmodel', MASK_ALGORITHM)
```
### B. Data Generators
In this section of the guide we will create a thread-safe image and mask generator for training models, for image segmentation. The primary reason for introducing thread safety is to allow multiple threads to consume batches from this generator without encountering data corruption. 
```
class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    From: https://github.com/keras-team/keras/issues/1638#issuecomment-338218517
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
    
def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg
```

```
@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):
   
    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0 # used to shuffle samples
    while True:
        
        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1 # keep a consistent order in shuffle


        for img_path, mask_path in zip(images_path, masks_path):

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []
```

This setup is for our machine learning workflow where batches of images and the corresponding masks are fed into a model during training. 

```
train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
```

NOTE: To fetch a batch to inspect
```
images, masks = next(train_generator)
print(f'Images shape: {images.shape}')  
print(f'Masks shape: {masks.shape}')
```
### C. Model Architecture
We will need to define a function `conv2d_block` to define a block of two 2D convolutional layers, for our CNN. That will be implemented as a Unet 
```
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
```







