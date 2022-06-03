# Denoising using Noise-to-Noise
## Abstract

In computed tomography (CT) imaging, image quality depends on the patient's exposure during the scan.
Reducing the exposure reduces patients' health risks, but also reduces image quality due to higher noise in the image.
A denoising technique preserving image features enables the acquisition of low-dose images without losing too much information.
In this work, the performance of convolutional neural networks in denoising CT images was evaluated.
    
The peculiarity of the network is that it does not require the usage of clean data for training.
instead, two or more independent noise realizations of each image are input during training(Noise2Noise).
The network output was compared with that of a network trained using clean images,
We will go through the results of works that indicate that the absence of clean images during training does not prevent the network from learning a good denoising model.

In this work, an analysis of the Noise2Noise learning strategy is done using real noise and synthetic datasets.
This paper demonstrates the effect of different factors on the ability to denoise CT images.
These factors includes:
- The depth of the encoder-decoder network.
- The different loss functions available and their parameters and even suggestion of a new loss function.
- The effect of learning rate scheduler on the network.
- The effect of the size of each epoch without decreasing the size of the dataset.

Mainly this work will look into the concept of teaching the network one
thing in order for it to learn to do another thing, with the different materializing of the deep learning concept in this situation.

## Code structure
- n2n = Network package
  - batchHistory.py = Tensorflow network callback to save history every batch.
  - dataset.py = Dataset - Read / Preprocess / Augment.
  - history_watcher.py = Tensorflow network callback to save history as CSV format to save history between runs.
  - model.py = Network model.
  - sqrtMSE.py = Tensorflow loss based on the MSE but with squared root over the result.
  - use_model.py = Using the model - Denoise images based on pretrained network.
  - train.py = Training the model - creating network weights for the testing\using process.
- util = Utilities for the project I/O
  - CsvReader.py = CsvReader with object convergence.
  - progressIterator.py = Check progress of iterator for printing.
- datasetio.py = Read the dataset from the nih website.
- LICENSE
- plot_history.py = Plot the results of the network.
- README.md


## Run Example

### Training
#### Program Arguments
```commandline
usage: train.py [-h] --image_dir IMAGE_DIR --image_size IMAGE_SIZE --test_dir
                TEST_DIR [--batch_size BATCH_SIZE]
                [--network_depth NETWORK_DEPTH] [--nb_epochs NB_EPOCHS]
                [--lr LR] [--disable_lr_Sched] [--steps STEPS]
                [--val_steps VAL_STEPS] [--loss LOSS]
                [--huber_loss_delta HUBER_LOSS_DELTA] [--noise_std NOISE_STD]
                [--weight WEIGHT] [--output_path OUTPUT_PATH]
                [--infinite_dataset]
train noise2noise model
optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        train image dir (default: None)
  --image_size IMAGE_SIZE
                        image size (default: None)
  --test_dir TEST_DIR   test image dir (default: None)
  --batch_size BATCH_SIZE
                        batch size (default: 16)
  --network_depth NETWORK_DEPTH
                        encoder-decoder network depth (default: 4)
  --nb_epochs NB_EPOCHS
                        number of epochs (default: 100)
  --lr LR               learning rate (default: 0.01)
  --disable_lr_Sched    disable learning rate scheduler (default: False)
  --steps STEPS         steps per epoch (default: None)
  --val_steps VAL_STEPS
                        steps per validation epoch (default: None)
  --loss LOSS           loss; mse', 'mae', or 'huber' is expected (default:
                        mse)
  --huber_loss_delta HUBER_LOSS_DELTA
                        huber loss delta parameter (default: 1.5)
  --noise_std NOISE_STD
                        gaussian noise standard deviation (default: 0.015)
  --weight WEIGHT       weight file for restart (default: None)
  --output_path OUTPUT_PATH
                        checkpoint dir (default: checkpoints)
  --infinite_dataset    make the dataset infinite (default: False)
```
#### Train example
```commandline
from n2n.train import main
import tensorflow as tf

depth = 4
batch = 8
epoches = 400
lr = 0.0001
loss = "huber"
huber_delta = 3.0

folder_name = f"d{depth}_b{batch}_e{epoches}_lr{lr}_{loss}_{huber_delta}"


gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
  main(
    f"--huber_loss_delta={huber_delta}",
    "--infinite_dataset",
    "--val_steps=30",
    "--steps=100",
    "--image_dir=dataset",
    "--test_dir=val",
    "--image_size=512", 
    f"--batch_size={batch}",
    f"--lr={lr}",
    f"--loss={loss}",
    f"--output_path={drive_path+folder_name}",
    f"--nb_epochs={epoches}"
    )
```


### Testing\Usage
#### Program Arguments
```commandline
usage: use_model.py [-h] --image_dir IMAGE_DIR [--network_depth NETWORK_DEPTH]
                    --weight_file WEIGHT_FILE [--output_dir OUTPUT_DIR]
Test trained model
optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        test image dir (default: None)
  --network_depth NETWORK_DEPTH
                        encoder-decoder network depth (default: 4)
  --weight_file WEIGHT_FILE
                        trained weight file (default: None)
  --output_dir OUTPUT_DIR
                        if set, save resulting images otherwise show result
                        using imshow (default: None)
```
#### Testing/Usage example
```commandline
from n2n.use_model import main
import tensorflow as tf

folder_name = "results"

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
  main(
    "--weight_file=/content/drive/MyDrive/d4_b8_e400_lr0.0001_huber_3.0/weights.382-30.732-46.93232.hdf5",
    "--image_dir=input",
    f"--output_dir={drive_path+folder_name}"
     )
```
