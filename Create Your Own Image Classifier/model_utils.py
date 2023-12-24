# model_utils.py
import torch
from torchvision import models
from torch import nn, optim


def build_model(arch="vgg16", hidden_units=512, learning_rate=0.001):
    # Load a pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1),
    )
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_model(
    model, train_loader, valid_loader, criterion, optimizer, epochs, use_gpu=False
):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                validation_loss += batch_loss.item()

                ps = torch.exp(outputs)
                equality = labels.data == ps.max(dim=1)[1]
                accuracy += equality.type(torch.FloatTensor).mean()

        # Print training and validation statistics
        print(
            f"Epoch {epoch + 1}/{epochs}.. "
            f"Train loss: {running_loss / len(train_loader):.3f}.. "
            f"Validation loss: {validation_loss / len(valid_loader):.3f}.. "
            f"Validation accuracy: {accuracy / len(valid_loader):.3f}"
        )

    print("Training complete!")


def save_checkpoint(
    model, arch, hidden_units, learning_rate, epochs, optimizer, class_to_idx, save_dir
):
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
    }

    checkpoint_path = f"{save_dir}/checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(args):
    checkpoint = torch.load(args.checkpoint)

    model = build_model(
        checkpoint["arch"], checkpoint["hidden_units"], checkpoint["learning_rate"]
    )[0]
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(
        model.classifier.parameters(), lr=checkpoint["learning_rate"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class, optimizer


def predict(image, model, idx_to_class, topk=1, use_gpu=False):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        image = torch.from_numpy(image).unsqueeze(0).float().to(device)
        output = model(image)

        probabilities, indices = torch.topk(
            torch.nn.functional.softmax(output[0], dim=0), topk
        )
        top_classes = [idx_to_class[idx.item()] for idx in indices]

    return probabilities.cpu().numpy(), top_classes
