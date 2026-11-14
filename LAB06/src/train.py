import os
import yaml
import mlflow
import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from utils import AttentionModel


# Load YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Reproducibility
torch.manual_seed(config["seed"])

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root=config["dataset"]["data_dir"],
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root=config["dataset"]["data_dir"],
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["training"]["batch_size"]
)

# Model
model = AttentionModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)

# MLflow
mlflow.set_experiment(
    config["mlflow"]["experiment_name"]
)

with mlflow.start_run():

    mlflow.log_params({
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "seed": config["seed"]
    })

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss = {running_loss:.4f}"
        )

        mlflow.log_metric(
            "loss",
            running_loss,
            step=epoch
        )

    # Evaluation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    print(f"Accuracy: {accuracy:.2f}%")

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # Save checkpoint
    torch.save(
        model.state_dict(),
        "../checkpoints/model.pth"
    )

    mlflow.log_artifact(
        "../checkpoints/model.pth"
    )