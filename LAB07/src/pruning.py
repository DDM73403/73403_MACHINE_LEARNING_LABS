import copy
import torch, torch.nn as nn
import torch.nn.functional as F
from common import *

SPARSITIES = [0, 0.25, 0.5, 0.75, 0.9, 0.95]
STRUCT_FRACS = [0.10, 0.25, 0.50]


def prunable(m):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]


def global_masks(m, s):
    """One threshold across all weights of all layers."""
    if s == 0:
        return {n: torch.ones_like(mod.weight) for n, mod in prunable(m)}
    allw = torch.cat([mod.weight.detach().abs().flatten() for _, mod in prunable(m)])
    thr = allw.kthvalue(max(1, int(s * allw.numel()))).values
    return {n: (mod.weight.detach().abs() > thr).float() for n, mod in prunable(m)}


def layer_masks(m, s):
    """Same sparsity s inside every layer (separate threshold per layer)."""
    masks = {}
    for n, mod in prunable(m):
        w = mod.weight.detach().abs()
        k = int(s * w.numel())
        masks[n] = torch.ones_like(w) if k == 0 else (w > w.flatten().kthvalue(k).values).float()
    return masks



def apply_masks(m, masks):
    with torch.no_grad():
        for n, mod in prunable(m):
            mod.weight.mul_(masks[n])


def weight_sparsity(m):
    tot = sum(mod.weight.numel() for _, mod in prunable(m))
    zero = sum((mod.weight == 0).sum().item() for _, mod in prunable(m))
    return zero / tot


def per_layer_sparsity(m):
    return {n: round((mod.weight == 0).float().mean().item(), 3) for n, mod in prunable(m)}


def finetune(m, data, epochs, lr=3e-4, masks=None, bs=128):
    """Fine-tune; if masks given, they are re-applied after every step (mask fixed)."""
    Xtr, ytr, _, _ = data
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        for xb, yb in batches(Xtr, ytr, bs, True):
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if masks is not None:
                apply_masks(m, masks)
        print(f"  ft epoch {ep+1}/{epochs}  acc {accuracy(m, data):.2f}%", flush=True)
    return m


def structured_prune(m, frac):

    """Remove the `frac` lowest-L2-norm filters of every conv layer and build a smaller net."""
    keeps = []
    for conv in (m.conv1, m.conv2, m.conv3):
        norms = conv.weight.detach().flatten(1).norm(dim=1)
        k = max(1, int(round(conv.out_channels * (1 - frac))))
        keeps.append(norms.topk(k).indices.sort().values)
    new = CNN(tuple(len(k) for k in keeps), m.h).to(DEV)
    prev = torch.tensor([0], device=DEV)          # conv1 has 1 input channel
    with torch.no_grad():
        for old, nw, keep in zip((m.conv1, m.conv2, m.conv3), (new.conv1, new.conv2, new.conv3), keeps):
            nw.weight.copy_(old.weight[keep][:, prev]); nw.bias.copy_(old.bias[keep]); prev = keep
        cols = (prev[:, None] * 9 + torch.arange(9, device=DEV)).flatten()   # 3x3 spatial per channel
        new.fc1.weight.copy_(m.fc1.weight[:, cols]); new.fc1.bias.copy_(m.fc1.bias)
        new.fc2.load_state_dict(m.fc2.state_dict())
    return new