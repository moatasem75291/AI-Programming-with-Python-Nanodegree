# train.py
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from model_utils import build_model, train_model, save_checkpoint
from data_utils import load_data


def main():
    parser = argparse.ArgumentParser(
        description="Train a new network on a dataset and save the model as a checkpoint."
    )

    parser.add_argument("data_directory", type=str, help="Path to the data directory")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        help="Choose architecture (vgg16, resnet18, etc.)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Set learning rate"
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Set number of hidden units in the classifier",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Set number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    # Load data
    train_loader, valid_loader, test_loader, class_to_idx = load_data(
        args.data_directory
    )

    # Build model
    model, criterion, optimizer = build_model(
        args.arch, args.hidden_units, args.learning_rate
    )

    # Train model
    train_model(
        model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.gpu
    )

    # Save checkpoint
    save_checkpoint(
        model,
        args.arch,
        args.hidden_units,
        args.learning_rate,
        args.epochs,
        optimizer,
        class_to_idx,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
