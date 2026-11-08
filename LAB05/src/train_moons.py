import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(
    n_samples=1000,
    noise=0.2,
    random_state=42
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1,1)

model = nn.Sequential(
    nn.Linear(2,8),
    nn.ReLU(),
    nn.Linear(8,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(100):
    optimizer.zero_grad()

    pred = model(X)

    loss = criterion(pred,y)

    loss.backward()

    optimizer.step()

print("Final loss:", loss.item())