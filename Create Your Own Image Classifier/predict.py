# predict.py
import argparse
import torch
from torchvision import models
from model_utils import load_checkpoint, predict
from data_utils import process_image, load_category_names


def main():
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image with the probability of that name."
    )

    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "--top_k", type=int, default=1, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Path to the mapping of categories to real names",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load checkpoint
    model, class_to_idx, idx_to_class, optimizer = load_checkpoint(args)

    # Process image
    processed_image = process_image(args.image_path)

    # Predict
    probs, classes = predict(processed_image, model, idx_to_class, args.top_k, args.gpu)

    # Load category names
    cat_to_name = load_category_names(args.category_names)

    # Map classes to real names
    class_names = [cat_to_name[idx] for idx in classes]

    # Print results
    for i in range(len(class_names)):
        print(f"Prediction {i+1}: {class_names[i]} with probability: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
