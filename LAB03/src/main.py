import torch
import torch.nn as nn
import time

from torchinfo import summary
from thop import profile



class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)



class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.embedding = nn.Linear(784, 128)

        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):

        x = self.flatten(x)

        x = self.embedding(x)

        x = x.unsqueeze(1)

        attn_output, _ = self.attention(x, x, x)

        x = attn_output.squeeze(1)

        x = self.classifier(x)

        return x



model = MLP()   




total_params = sum(p.numel() for p in model.parameters())

print("\nTotal Parameters:", total_params)


summary(model, input_size=(1, 1, 28, 28))



dummy_input = torch.randn(1, 1, 28, 28)

flops, params = profile(
    model,
    inputs=(dummy_input,),
    verbose=False
)

print("\nMeasured FLOPs:", flops)
print("Measured Parameters:", params)


start = time.perf_counter()

for _ in range(100):
    _ = model(dummy_input)

end = time.perf_counter()

latency = ((end - start) / 100) * 1000

print("Average Latency (ms):", latency)