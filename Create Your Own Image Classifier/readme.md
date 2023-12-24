# Image Classifier Project

Welcome to the Image Classifier project! This project is built using PyTorch and torchvision to create a powerful image classifier. The classifier can be trained on a dataset and then used to make predictions on new images.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Project Structure](#project-structure)
- [Additional Notes](#additional-notes)

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- NumPy

### Installation

Clone the repository:

```bash
https://github.com/moatasem75291/AI-Programming-with-Python-Nanodegree/tree/main/Create%20Your%20Own%20Image%20Classifier
```

## Usage

### Training the Model

To train the model on a custom dataset, follow these steps:

1. Organize your dataset into three folders: `train`, `valid`, and `test`.

2. Update the `data_directory` argument in `train.py` with the path to your dataset.

```bash
python train.py --data_directory folder_path_that_contains_your_dataset
```

3. The trained model will be saved in the `saved_models` directory as a checkpoint file.

### Making Predictions

To make predictions on a new image using the trained model, follow these steps:

1. Update the `image_path` and `checkpoint` arguments in `predict.py` with the path to the image and checkpoint file, respectively.

```bash
python predict.py image.jpg --checkpoint saved_models/checkpoint.pth
```

2. Optionally, use the `--top_k` and `--gpu` flags to specify the number of top predictions and whether to use GPU for inference.

## Project Structure

- `data_utils.py`: Contains functions for loading and processing image data.

- `model_utils.py`: Defines functions for building, training, and saving/loading the model.

- `predict.py`: Script for making predictions on new images.

- `train.py`: Script for training a new network on a dataset and saving the model as a checkpoint.

## Additional Notes

- You can customize various parameters such as architecture, learning rate, hidden units, etc., by providing additional command-line arguments to the scripts.
- For more details on available options, use the `--help` flag with each script.
