import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)

from utils import AttentionModel


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128
)


def evaluate(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    return 100 * correct / total


experiments = [

    ("Adam", "StepLR"),
    ("Adam", "CosineAnnealingLR"),
    ("Adam", "ReduceLROnPlateau"),

    ("SGD", "StepLR"),
    ("SGD", "CosineAnnealingLR"),
    ("SGD", "ReduceLROnPlateau")
]

for optimizer_name, scheduler_name in experiments:

    print(f"\n{optimizer_name} + {scheduler_name}")

    model = AttentionModel()

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

    else:

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

    if scheduler_name == "StepLR":

        scheduler = StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

    elif scheduler_name == "CosineAnnealingLR":

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=10
        )

    else:

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min"
        )

    for epoch in range(10):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(running_loss)
        else:
            scheduler.step()

    accuracy = evaluate(model)

    print(
        f"Accuracy = {accuracy:.2f}%"
    )