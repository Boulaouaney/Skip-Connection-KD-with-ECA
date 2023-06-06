# ResNet Skip-Connection Knowledge Distillation with Efficient Channel Attention

## Table of Contents

1. [About](#about)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installing](#installing)
3. [Usage](#usage)
   - [Training](#training)
   - [Visualization](#visualization)

## About

This project, stemming from a Master's Thesis, presents an innovative method to enhance Convolutional Neural Networks (CNNs) performance. It integrates a modified skip-connection knowledge distillation technique with an efficient channel attention module, aiming to boost the CNNs' efficiency and performance while reducing their size and computational complexity.

Tested on the CIFAR-10 dataset, our method demonstrated significant performance improvements in both large teacher-small student and self-distillation scenarios, alongside a notable reduction in model sizes. It holds potential for real-world applications and for testing on diverse datasets, thereby moving towards more efficient CNNs for image classification tasks.

Alongside the main implementation, this project includes the visualization of CNN's intermediate layer outputs. It provides a glimpse into the network's learning process and model's internal decision-making. Although not the main focus, this visualization aids in understanding the transformations within the model, contributing to transparency and potential enhancements in the CNN's design.

Also included is a script for plotting the training metrics such as accuracy and loss for both teacher and student models (found in [numpy_outputs/plotting.py](./numpy_outputs/plotting.py)). The visualization script uses saved training metrics, which are stored as numpy arrays, to generate interactive plots.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project was tested on a Windows 10 machine with CUDA 11.6 and Python >=3.8

### Installing

1. Ensure that you have CUDA 11.6 installed. This is crucial for running the PyTorch and torchvision libraries.

    [Download the CUDA 11.6 version corresponding to your system here.](https://developer.nvidia.com/cuda-downloads)


2. Open a terminal and clone this repo on your local machine.

```
git clone https://github.com/Boulaouaney/Skip-Connection-KD-with-ECA.git
cd Skip-Connection-KD-with-ECA
```
3. Create a Python virtual environment and activate it. This is an optional but recommended step to avoid any conflicts with packages in your global Python environment.

```
python -m venv env
.\env\Scripts\activate
```
4. Install PyTorch 1.12.0 (version tested with the project).

```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
5. Install the project dependencies using the following command.

```
pip install -r requirements.txt
```
## Usage

These are some of the main scripts of the project. For most of the scripts you can check the file for usage or run 
```
python <script_name>.py --help
```
### Training
#### [model_training.py](./model_training.py)
Script to train base ResNet model without Knowledge Distillation or ECA.

#### [model_training_mid.py](./model_training_mid.py)
Script to train a ResNet teacher model then perform knowledge distillation to a student model.

Trained models will be saved to [saved_pth_model](./saved_pth_model/). And their corresponding training metrics will be saved to [numpy_outputs](./numpy_outputs/) as numpy arrays. You can also track training metrics with tensorboard:
```
tensorboard --logdir runs
```
### Visualization
#### [features_visualization.ipynb](./features_visualization.ipynb)
This notebook is a step-by-step guide to visualize intermediate layer outputs of a pre-trained model.

#### [visualize.py](./visualize.py)
This script uses `torchviz` to save a png of the visualized architecture of a pytorch .pth model.

#### [convertToOnnx.py](./convertToOnnx.py)
You can use this script to convert the pretrained pytorch model to ONNX format. I used it mainly to visualize the model with [Netron](https://github.com/lutzroeder/netron) since it has  better support for ONNX

#### [numpy_outputs/plotting.py](./numpy_outputs/plotting.py)
This script can be used to to plot the training metrics of the teacher and student models. The script plots either accuracy or loss (specified by the user), leveraging numpy arrays saved during model training.