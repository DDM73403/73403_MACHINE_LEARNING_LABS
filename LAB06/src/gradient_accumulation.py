import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

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
    batch_size=64
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64
)

model = AttentionModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

accumulation_steps = 4

for epoch in range(5):

    model.train()

    optimizer.zero_grad()

    for i, (images, labels) in enumerate(train_loader):

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:

            optimizer.step()

            optimizer.zero_grad()

    print(f"Epoch {epoch+1} complete")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(
    f"Gradient Accumulation Accuracy = {accuracy:.2f}%"
)
