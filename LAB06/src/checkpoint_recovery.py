import torch

from utils import AttentionModel


model = AttentionModel()

checkpoint_path = "../checkpoints/model.pth"

model.load_state_dict(
    torch.load(checkpoint_path)
)

model.eval()

print("Checkpoint loaded successfully")
print("Training resumed from saved model")
