# CosmiQ_SN4_Baseline
[![GitHub release](https://img.shields.io/github/release/cosmiq/Cosmiq_SN4_Baseline.svg)](https://GitHub.com/cosmiq/CosmiQ_SN4_Baseline/releases/)
![docker](https://img.shields.io/docker/build/cosmiqworks/cosmiq_sn4_baseline.svg)
![license](https://img.shields.io/github/license/cosmiq/CosmiQ_SN4_Baseline.svg)
[![tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Check%20out%20the%20CosmiQ%20SpaceNet%204%20Baseline%20Model%20on%20GitHub%20and%20Medium!%20https://github.com/cosmiq/cosmiq_sn4_baseline&url=https://medium.com/the-downlinq/a-baseline-model-for-the-spacenet-4-off-nadir-building-detection-challenge-6b7983312b4b)
### Baseline model and utilities for [SpaceNet Challenge Round 4: Off-Nadir Building Footprint Detection](https://www.topcoder.com/spacenet)

This repository contains code to train baseline models to identify buildings in the SpaceNet 4: Off-Nadir Building Footprint Detection Challenge, and then use those models to generate predictions for test imagery in the competition format. See [the DownLinQ post about the baseline](https://medium.com/the-downlinq/a-baseline-model-for-the-spacenet-4-off-nadir-building-detection-challenge-6b7983312b4b) for more information.

### Table of Contents:
- [Requirements](#requirements)
- [Repository contents](#repository-contents)
- [Installation](#installation)
- [Usage](#usage)
---
## Requirements
- Python 3.6
- The SpaceNet 4 training and test datasets. See download instructions under "Downloads" [here](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17313&pm=15148). These datasets are freely available from an AWS S3 bucket, and are most easily downloaded using the AWS CLI. The dataset comprises about 75 GB of gzipped tarballs of imagery data with accompanying .geojson-formatted labels. __You must download and expand the tarballs prior to running any of the scripts in the bin directory or performing any model training with the cosmiq_sn4_baseline python module.__
- Several python packages (rasterio, tensorflow, keras, opencv, spacenetutilities, numpy, pandas, scikit-image). All are installed automatically during [Installation](#installation) using the Dockerfile or `pip`.

#### Optional dependencies:
- NVIDIA GPUs (see notes under Dockerfile and pip in [Installation](#installation))
- Tensorflow-GPU
- Tensorboard (for live monitoring of model training)
- nvidia-docker 2
---
## Repository contents
- __cosmiq_sn4_baseline directory, setup.py, and MANIFEST.in__: Required components of the pip-installable cosmiq_sn4_baseline module. Components of that module:
  - __models__: Keras model architecture for model training.
  - __utils__: Utility functions for converting spacenet geotiff and geojson data to the formats required by this library.
  - __DataGenerator__: `keras.utils.Sequence` subclasses for streaming augmentation and data feeding into models.
  - __losses__: Custom loss functions for model training in Keras.
  - __metrics__: Custom metrics for use during model training in Keras.
  - __callbacks__: Custom callbacks for use during model training in Keras.
  - __inference__: Inference code for predicting building footprints using a trained Keras model.
- __Dockerfile__: [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) Dockerfile to create images and containers with all requirements to use this package.
- __bin directory__: Scripts for data pre-processing, model creation, model training, and running inference using `cosmiq_sn4_baseline`.

---
## Installation

#### Docker container setup (NVIDIA GPU usage only)
Building the nvidia-docker image for this repository will install all of the package's dependencies, as well as the `cosmiq_sn4_baseline` python module, and will provide you with a working environment to run the data pre-processing, model training, and inference scripts (see [Usage](#usage)). To build the container:
1. Clone the repository: `git clone https://github.com/cosmiq_sn4_baseline.git`
2. `cd CosmiQ_SN4_Baseline`
3. `nvidia-docker build -t cosmiq_sn4_baseline .`
4. `NV_GPU=[GPU_ID] nvidia-docker run -it --name space_base cosmiq_sn4_baseline`  
Replace `[GPU_ID]` with the identifier for the GPU you plan to use.


If you do not have access to GPUs, you can still install the python codebase using `pip` and perform model training and inference; however, it will run much more slowly.
#### pip installation of the codebase

If you do not have access to GPUs, or you only wish to install the python library within an existing environment, the python package is installable via [pip](https://pypi.org/project/pip/) using two approaches:
1. Clone the repository and install locally  
  - Navigate to your desired destination directory in a shell, and run: `git clone https://github.com/cosmiq/cosmiq_sn4_baseline.git`
  - `cd CosmiQ_SN4_Baseline`
  - `pip install -e .`
2. Install directly from GitHub _(python module only)_
  - Within a shell, run `pip install -e git+git://github.com/cosmiq_sn4_baseline`
---
## Usage
The python library and scripts here can be used one of two ways:
1. Use the scripts within the __bin__ directory for data pre-processing, model training, and inference.
2. Write your own data processing, training, and inference code using the classes and functions in the `cosmiq_sn4_baseline` module.

The second case is self-explanatory, and the various functions and classes' usage is documented in the codebase.

All of the scripts in __bin__ can be called from the command line with the format  
`python [script.py] [arguments]`  

The arguments are documented within the codebase. You can also receive a description of their usage by running  
`python [script.py] -h`  
from the command line. Their usage is also detailed below.

#### Command line functions
__make_np_arrays.py__ --dataset_dir (source_directory) --output_dir (destination_directory) [--verbose --create_splits]  

Convert imagery to a Keras model-usable format.
__NOTE__: IMAGERY TARBALLS MUST BE EXPANDED BEFORE THIS SCRIPT IS CALLED!  
_Arguments:_  
- `--dataset_dir`, `-d`: Path to the directory containing both the training and test datasets. The structure should be thus:  
```
dataset_dir
|
+-- SpaceNet-Off-Nadir_Train
|   +-- Unzipped imagery directories from tarballs (e.g. Atlanta_nadir29_catid_1030010003315300/)
|   +-- geojson/  # directory containing building labels
|
+-- SpaceNet-Off-Nadir_Test
    +-- Unzipped imagery directories from tarballs
```
- `--output_dir`, `-o`: Path to the desired directory to save output data to. Outputs will be comprised of 8-bit BGR .tifs, binary mask .tifs for each location chip, and NumPy arrays containing both of the above. The output structure will be thus when completed:
```
output_dir
|
+-- train_rgb: directory containing 8-bit BGR tiffs for each collect/chip pair from SpaceNet-Off-Nadir_Train subdirs
+-- masks: directory containing tiff binary masks for each chip from SpaceNet-Off-Nadir_Train/geojson
+-- test_rgb: directory containing 8-bit BGR tiffs for each collect/chip pair from SpaceNet-Off-Nadir_Test subdirs
+-- all_train_ims.npy: numpy array of training imagery
+-- all_test_ims.npy: numpy array of test imagery
+-- all_train_masks.npy: numpy array of binary building footprint masks
+-- training_chip_ids.npy: numpy array of chip IDs for each image in the training array
+-- test_chip_ids.npy: numpy array of chip IDs for each image in the test array
|  [Optional, see --create_splits flag below]
+-- nadir_train_ims.npy: numpy array of training imagery from nadir angles 7-25 only
+-- offnadir_train_ims.npy: numpy array of training imagery from nadir angles 26-40 only
+-- faroffnadir_train_ims.npy: numpy array of training imagery from nadir angles 41-53 only
+-- nadir_train_masks.npy
+-- offnadir_train_masks.npy
+-- faroffnadir_train_masks.npy
```
- `--verbose`, `-v`: Verbose text output while running? Defaults to `False`.
- `--create_splits`, `-s`: Make nadir, offnadir, and far-offnadir training subarrays? Defaults to `False`. Note that using this flag roughly doubles the disk space required for imagery numpy array storage.  

__train_model.py__ --data_path (source_directory) --output_path (path to model output) [--subset ['all', 'nadir', 'offnadir', or 'faroffnadir'] --seed (integer) --model ['ternausnetv1' or 'unet', see docstring] --tensorboard_dir (path to desired tensorboard log output dir)]

Train a model on the data. __Note__: Source imagery must be generated using make_np_arrays.py prior to use.  
_Arguments:_
- `--data_path`, `-d`: Path to the source dataset files. This corresponds to `--output_dir` from make_np_arrays.py.
- `--output_path`, `-o`: Path to save the trained model to. Should end in '.h5' or '.hdf5'.
- `--subset`, `-s`: Train on all of the data, or just a subset? Defaults to 'all'. Other options are 'nadir', 'offnadir', or 'faroffnadir'. To use the subset options, the imagery subsets must have been produced using the `--create_splits` flag in make_np_arrays.py.
- `--seed`, `-e`: Seed for random number generation in NumPy and TensorFlow. Alters initialization parameters for each model layer. Defaults to 1337.
- `--model`, `-m`: Options are 'ternausnetv1' (default) and 'unet'. See cosmiq_sn4_baseline.models for details on model architecture.
- `--tensorboard_dir`, `-t`: Destination directory for tensorboard log writing. Optional, only required if you want to use tensorboard to visualize training.

__make_predictions.py__ --model_path (path) --test_data_path (path to test_data/) --output_dir (desired directory for outputs) [--verbose --angle_set ('all', 'nadir', 'offnadir', or 'faroffnadir') --angle (integer angles) --n_chips (integer number of unique chips to predict for each angle) --randomize_chips (randomize the order of chips before subsetting) --footprint_threshold (integer minimum number of pixels for a footprint to be kept) --window_step (integer number of pixels to step in x,y directions during inference)]
make_predictions.py runs inference on the test image set. It scans across the X,Y axes of each images in step size (`--window_step`, defaults to 64), producing overlapping predictions, and then averages the predictions. This helps reduce edge effects.
_Arguments:_  
- `--model_path`, `-m`: path to the .hdf5 or .h5 file saved by train_model.py.
- `--test_data_path`, `-t`: Path to the test_data directory produced by make_np_arrays.py.
- `--output_dir`, `-o`: path to the desired output directory to save data to. Defaults to test_output in the current working directory. The directory structure will be thus:
```
output_dir
|
+-- output_geojson: directory containing geojson-formatted footprints for visual inspection (i.e. with QGIS)
|   +-- subdir for each nadir angle
|
+-- predictions.csv: SpaceNet challenge-formatted .csv for passing to competition evaluation
```
- `--verbose`, `-v`: produce verbose text output? Defaults to `False`.
- `--angle_set`, `-as`: Should inference be run on imagery from all angles, or only a subset? Options are ['all', 'nadir', 'offnadir', and 'faroffnadir']. Not to be used at the same time as `--angle`.
- `--angle`, `-a`: Specific angle[s] to produce predictions for. Not to be used at the same time as `--angle_set`.
- `--n_chips`, `-n`: Number of unique chips to run evaluation on. Defaults to all.
- `--randomize_chips`, `-r`: Randomize order of chips prior to subsetting? Defaults to `False`. Only has an effect if using `--n_chips`.
- `--footprint_threshold`, `-ft`: Minimum footprint size to be kept as a building prediction. Defaults to 0.
- `--window_step`, `-ws`: Step size for sliding window during inference. Defaults to 64.
- `--simplification_threshold`, `-s`: Threshold, in meters, for simplifying polygon vertices. Defaults to 0 (no simplification). Failure to simplify can result in _very_ large .csv outputs (>500 MB), which are too big to use in submission for the competition.

Good luck in SpaceNet 4: Off-Nadir Buildings!
