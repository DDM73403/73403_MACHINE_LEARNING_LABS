import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

from torch.profiler import profile
from torch.profiler import ProfilerActivity

from utils import AttentionModel


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

model = AttentionModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True
) as prof:

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        break

print(
    prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20
    )
)