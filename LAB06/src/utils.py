import torch
import torch.nn as nn


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