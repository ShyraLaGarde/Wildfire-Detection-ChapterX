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
  - [C. Convolution Layers](#C-Convolution-Layers)
  - [D. Build Model](#D-Build-Model)
    - [i. Model Archeticture](i-Model-Archeticture)
    - [ii. Create and Build Model](ii-Create-and-Build-Model)
    - [iii. Parameter Selection](iii-Parameter-Selection)
    - [iv. GPU Configuration](#iv-GPU-Configuration)
    - [v. Callbacks](#v-Training-Callbacks)
  - [E. Model Training](#E-Model-Training)
    - [i. Plot Results (without WandB)](#i-Plot-Results)
  * [Inference](#Inference)
  * [Metrics](#Metrics)
   
  

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
      
In this script we are interested in the Kumar-Roy study 
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
Our model was developed for wildfire detection over the Canada Region using a U-Net architecture, a popular choice for image segmentation task. This part of the guide we guide you through the steps from parameter selection to model building and training.

### A. Model Set Up
We will set an output directory to store our model results, initally we evaluate the training and validation accuarcy and loss. We will normilize our 256x256 pixel images and define the hyperparameters used to train our model. 
```
OUTPUT_DIR = './train_output/'

MAX_PIXEL_VALUE = 65535 # Max. pixel value, used to normalize the image
IMAGE_SIZE = (256, 256)

MASK_ALGORITHM = 'Kumar-Roy'

WORKERS = 4
# Early stopping patience threshold
EARLY_STOP_PATIENCE = 5
# Save model checkpoints every 5 epochs
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
By introducing thread safety into the data loading process, we load images and corresponding segmentation masks and normalize the image data. The function get_img_arr handles 10-band images, while get_img_762bands loads only specific bands for input. For this experiment our wildfire detection model input will consider the SWIR 1 (band 6), SWIR 2 (band 7), and Blue (band 2).

Next we will create a function to generate batches of images and masks. It shuffles the data each epoch for more robust training, ensuring thread-safe data loading. The shuffle flag and `random_state=42` allow for reproducibility in shuffling.
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
### C. Convolution Layers
We will need to define a function `conv2d_block` to define a block of two 2D convolutional layers, for our UNet architecture. Each block consists of two convolutional layers followed by batch normalization and ReLU activation. 

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
### D. Build Model 
This section builds the U-Net model for wildfire detection. We stack several conv2d_blocks in both the (down-sampling) and the expansive path (up-sampling)
To evaluate our experiments we employ Weights & Biases (WandB) functions. To use these resourse we will ave to install WandB and create an account. Follow the link here for instructions on how to use WandB

#### i. Model Architecture 
The architecture size of the U-Net models defined can be analyzed based on several factors: the number of filters, the input image size, the number of layers, and the operations within each layer.
  - Input: input_img with shape (256, 256, 10)
  - n_filters increases by a factor of 2 after each MaxPoolong layer
```
def get_unet(nClasses, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    input_img = Input(shape=(input_height,input_width, n_channels))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
```
#### ii. Create and Build Model 
We next create a function that will create our U-Net model. The parameters are passed in as arguments, the model reference a configuration dictionary (CFG) that allows the model to be customized. 
```
def get_model(model_name='unet', nClasses=1, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):   
    return model(
            nClasses      = nClasses,  
            input_height  = input_height, 
            input_width   = input_width,
            n_filters     = CFG['n_filters'],
            dropout       = CFG['dropout'],
            batchnorm     = batchnorm,
            n_channels    = CFG['n_channels']
        )
```
Next we create a function to build the model by calling get_model and passing the relevant parameters. After building the model, it compiles it using the Adam optimizer and sets the loss function to binary cross-entropy suitable for our binary classification task. 
```
def build_model():
    #define model
    model = get_model(MODEL_NAME, 
                  input_height=IMAGE_SIZE[0], 
                  input_width=IMAGE_SIZE[1], 
                  n_filters=N_FILTERS, 
                  n_channels=N_CHANNELS)

    model.compile(optimizer = Adam(), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    return model
```
#### iii. Parameter Selection 
The CFG dictionary that holds key hyperparameters like the number of filters, channels, batch size, and epochs for training. In this step we will also initialize a WandB run for tracking the model's performance and hyperparameters during training.
```
CFG = dict(
    n_filters = 16,
    n_channels = 3, 
    epochs = 50,
    batch_size = 64,
    dropout = 0.1,   
)

MODEL_NAME = 'unet'
# Build model
model = build_model()

# Initialise run
run = wandb.init(project = 'WFD-Kumar-Roy',
                 config = CFG,
                 save_code = True,
                 name = 'Base_Metrics',
)
```
Alternatively id you do not have a WandB account set up we can  plot the training and validation curves after training by setting the `PLOT_HISTORY = True `.
```
# if True plot the training and validation graphs
PLOT_HISTORY = True 
# Schoeder, Murphy or Kumar-Roy
MASK_ALGORITHM = 'Kumar-Roy'

MODEL_NAME = 'unet'
RANDOM_STATE = 42
IMAGES_DATAFRAME = '/Users/<Local>/WildfireDetection/Data/images_masks.csv'
```

We will need to read a CSV file containing image and mask paths into a pandas DataFrame. This DataFrame will be used later for data generation during model training.

```
df = pd.read_csv(IMAGES_DATAFRAME, header=None, names=['images', 'masks'])
```

#### iv. GPU Configuration 
We will next configure GPU settings and handle checkpointing.
```
# If not zero will be load as weights
INITIAL_EPOCH = 0
RESTART_FROM_CHECKPOINT = None
if INITIAL_EPOCH > 0:
    RESTART_FROM_CHECKPOINT = os.path.join(OUTPUT_DIR, 'checkpoint-{}-{}-epoch_{:02d}.hdf5'.format(MODEL_NAME, MASK_ALGORITHM, INITIAL_EPOCH))

FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format('CNNmodel', MASK_ALGORITHM)

CUDA_DEVICE = 1

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass
```
#### v. Training Callbacks
A callback to log model metrics and save the model during training. Our overfiiting protocols include early stopping if the validation loss does not improve and the model saves checkpoints at regular intervals. 

```
wandb_callback = WandbCallback(
    monitor='val_loss', 
    mode='min', 
    save_model=True
)

es = EarlyStopping(monitor='val_loss', 
                   mode='min', 
                   verbose=1, 
                   patience=EARLY_STOP_PATIENCE)

checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME),
                             monitor='loss', 
                             verbose=1,
                             save_best_only = True, 
                             mode='auto',
                             period=CHECKPOINT_PERIOD)
```
### E. Model Training 
To finish our modeling process we have to train our model. The model is trained using the generator we created `train_generator` the validation data is passed using `validation_generator`. Training is tracked and controlled by the callback we defined earlier. 
```
if INITIAL_EPOCH > 0:
    model.load_weights(RESTART_FROM_CHECKPOINT)

print('Training using {}...'.format(MASK_ALGORITHM))
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // CFG['batch_size'],
    validation_data=validation_generator,
    validation_steps=len(images_validation) // CFG['batch_size'],
    callbacks=[wandb_callback, checkpoint],
    epochs= CFG['epochs'],
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
)
print('Train finished!')
```
After Traing the model we will want to save our weights.
```
print('Saving weights')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("Weights Saved: {}".format(model_weights_output))
```
#### i. Plot Results 
To visualize our results without WandB we will use the following scripts
```
def plot_history(history, out_dir):
    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.savefig(os.path.join(out_dir, "accuracyWANDBS1.png"), dpi=300, bbox_inches='tight')
    plt.clf()
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.savefig(os.path.join(out_dir, "lossWANDBS1.png"), dpi=300, bbox_inches='tight')
    plt.clf()
```

```
if PLOT_HISTORY:
    plot_history(history, OUTPUT_DIR)
```
# Inference 
At this point in the guide we will next load the test images and masks, run predictions using our pre-trained model, and save the results for further analysis. This step incorporates useful features like progress tracking, error handling, and saving both the ground truth and predicted masks.

This process begins with setting the timing and CPU for model execution, defining output directories to save the results and logs made during our inference. 

We will use pre-trained model weights and apply a 0.25 fire threshold, therefore any prediction value above this threshold is classified as a fire pixel. 
```
start = time.time()

CUDA_DEVICE = 0

OUTPUT_DIR = './log'
OUTPUT_CSV_FILE = 'output_v1_{}_{}.csv'.format(MODEL_NAME, MASK_ALGORITHM)
WRITE_OUTPUT = True

WEIGHTS_FILE = './train_output/model_{}_{}_final_weights.h5'.format(MODEL_NAME, MASK_ALGORITHM)

TH_FIRE = 0.25
```

To ensure reproducible inference results we set a output directory for saving both prediction and ground truth arrays. `CUDA_VISIBLE_DEVICES` ensures TensorFlow uses the specified GPU for inference. GPU configuration allow dynamic memory allocation. 
Finally we reset the random bit generator to prevent conflicts that may arise in the random number generation. 

```
if not os.path.exists(os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays')):
    os.makedirs(os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays'))

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
    np.random.bit_generator = np.random._bit_generator
except:
    pass
```

Our initial step in Inference is to read in CSV files that contain the file paths to test images and their corresponding ground truth masks.

```
IMAGES_CSV = ''/Users/local/WildfireDetection/Data/images_test.csv'
MASKS_CSV = ''/Users/local/WildfireDetection/Data/masks_test.csv'

images_df = pd.read_csv(IMAGES_CSV)
masks_df = pd.read_csv(MASKS_CSV)

print('Loading images...')
images = []
masks = []

images = [ os.path.join(IMAGES_PATH, image) for image in images_df['images'] ]
masks = [ os.path.join(MASKS_PATH, mask) for mask in masks_df['masks'] ]
```
Then we initialize the Convolution Neural network model based on the specified architecture and configuration. The model is configured to match the input data in terms of image size and number of channels, ensuring compatibility during inference.

```
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.summary()
```
Note Our Model Architecture is the following: [IMAGE]
Next, we will load pre-trained weights into the model for inference.
```
WEIGHTS_FILE = '/Users/local/WildfireDetection/train_output/model_CNNmodel_Kumur-Roy_final_weights.h5'
print('Loading weghts...')
model.load_weights(WEIGHTS_FILE)
print('Weights Loaded')

print('# of Images: {}'.format( len(images)) )
print('# of Masks: {}'.format( len(masks)) )
```
To conclude the Inference section of the guide we use the following script to iterate through each image and its corresponding mask, perform inference using the model, and save both the ground truth and predicted outputs.
```
step = 0
steps = len(images)
for image, mask in zip(images, masks):
    
    try:
        
        # img = get_img_arr(image)
        img = get_img_762bands(image)
        
        mask_name = os.path.splitext(os.path.basename(mask))[0]
        image_name = os.path.splitext(os.path.basename(image))[0]
        mask = get_mask_arr(mask)

        txt_mask_path = os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'grd_' + mask_name + '.txt') 
        txt_pred_path = os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'det_' + image_name + '.txt') 

        y_pred = model.predict(np.array( [img] ), batch_size=1)

        y_true = mask[:,:,0] > TH_FIRE
        y_pred = y_pred[0, :, :, 0] > TH_FIRE


        np.savetxt(txt_mask_path, y_true.astype(int), fmt='%i')
        np.savetxt(txt_pred_path, y_pred.astype(int), fmt='%i')

        step += 1
        
        if step%100 == 0:
            print('Step {} of {}'.format(step, steps)) 
            
    except Exception as e:
        print(e)
        
        with open(os.path.join(OUTPUT_DIR, "error_log_inference.txt"), "a+") as myfile:
            myfile.write(str(e))
    

print('Done!')
```
# METRICS 
At this point in the guide we have Preprocessed data, Trained a UNet using the cleaned data, and generated Inferences on our model. The next part in the guide will provide functions used as key metrics and post-processing steps for evaluating the performance of our  model for wildfire detection. Each metric provides valuable insights into different aspects of the model’s performance, from overall accuracy to detailed region-based analysis.

Dice Coefficient:  is computed by dividing twice the intersection of the predicted `y_pred` and actual masks `y_true` by the sum of their areas. This metric is commonly used in image segmentation tasks to measure the similarity between these mask. 
```
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice
```
Pixel Accuracy: This function computes the pixel accuracy, measuring how many pixels in the predicted mask `y_pred` and actual masks `y_true`. Pixel accuracy is especially useful for evaluating how well the model performs in binary classification of pixels. 
```
def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)
 
    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy
```
The confusion matrix is fundamental for calculating additional metrics such as precision, recall, and F1-score.
```
## Confusion Matrix for True/False Negative/Positive 
# Scikit 
def statistics (y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

# data-centric approch
def statistics2 (y_true, y_pred):
    py_actu = pd.Series(y_true, name='Actual')
    py_pred = pd.Series(y_pred, name='Predicted')
    df_confusion = pd.crosstab(py_actu, py_pred)
    return df_confusion[0][0], df_confusion[0][1], df_confusion[1][0], df_confusion[1][1]

# manually
def statistics3 (y_true, y_pred):
    y_pred_neg = 1 - y_pred
    y_expected_neg = 1 - y_true

    tp = np.sum(y_pred * y_true)
    tn = np.sum(y_pred_neg * y_expected_neg)
    fp = np.sum(y_pred * y_expected_neg)
    fn = np.sum(y_pred_neg * y_true)
    return tn, fp, fn, tp
```
Jaccard Index: is computed as the ratio of the intersection of the predicted and actual masks to their union. The Jaccard Index tends to penalize false positives more, making it stricter for evaluating segmentation.
```
def jaccard3 (im1, im2):
    """
    Computes the Jaccard metric, a measure of set similarity.
    Parameters 
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    jaccard : float 
        Jaccard metric returned is a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
    Notes
    -----
    The order of inputs for `jaccard` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    jaccard_fire = intersection.sum() / float(union.sum())

    im1 = np.logical_not(im1)
    im2 = np.logical_not(im2)

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    jaccard_non_fire = intersection.sum() / float(union.sum())
    jaccard_avg = (jaccard_fire + jaccard_non_fire)/2

    return jaccard_fire, jaccard_non_fire, jaccard_avg
```
This next function labels connected components in a binary image, which helps identify distinct regions of fire in the predicted `y_pred` or ground truth mask `y_true`. The second function relabels regions in the predicted mask `y_pred` based on the ground truth `y_true`, ensuring that the predicted labels correspond to the true labels.
```
def connected_components (array):
    structure = np.ones((3, 3), dtype=np.int)  # 8-neighboorhood
    labeled, ncomponents = label(array, structure)
    return labeled

def region_relabel (y_true, y_pred):
    index = 0
    for p in y_true:
        if y_pred[index] == 1:
            if y_true[index] > 0:
                y_pred[index] = y_true[index]
            else:
                y_pred[index] = 999
        index += 1
```












