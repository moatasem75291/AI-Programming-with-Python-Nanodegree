# data_utils.py
import torch
import json
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


def load_data(data_directory):
    train_dir = data_directory + "/train"
    valid_dir = data_directory + "/valid"
    test_dir = data_directory + "/test"

    # Define data transformations
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_loader, valid_loader, test_loader, train_data.class_to_idx


def process_image(image_path):
    img = Image.open(image_path)

    img = img.resize((256, 256))
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))

    img = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))

    return img


def load_category_names(category_names_path):
    with open(category_names_path, "r") as f:
        cat_to_name = json.load(f)
    return cat_to_name
