import os, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS, RESULTS, FIGURES = OUT / "models", OUT / "results", OUT / "figures"
for d in (MODELS, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)


def load_data():
    root = str(ROOT / "data")
    tr = datasets.FashionMNIST(root, train=True, download=True)
    te = datasets.FashionMNIST(root, train=False, download=True)
    f = lambda d: ((d.data.float() / 255 - 0.2860) / 0.3530).unsqueeze(1).to(DEV)
    return f(tr), tr.targets.to(DEV), f(te), te.targets.to(DEV)


def batches(X, y, bs, shuffle):
    idx = torch.randperm(len(X), device=X.device) if shuffle else torch.arange(len(X), device=X.device)
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


class CNN(nn.Module):
    def __init__(s, c=(32, 64, 128), h=768):
        super().__init__()
        s.c, s.h = tuple(c), h
        s.conv1 = nn.Conv2d(1, c[0], 3, padding=1)
        s.conv2 = nn.Conv2d(c[0], c[1], 3, padding=1)
        s.conv3 = nn.Conv2d(c[1], c[2], 3, padding=1)
        s.fc1 = nn.Linear(c[2] * 9, h)   # 28 -> 14 -> 7 -> 3
        s.fc2 = nn.Linear(h, 10)

    def forward(s, x):
        for conv in (s.conv1, s.conv2, s.conv3):
            x = F.max_pool2d(F.relu(conv(x)), 2)
        return s.fc2(F.relu(s.fc1(x.flatten(1))))


@torch.no_grad()
def accuracy(m, data):
    _, _, X, y = data
    m.eval(); c = 0
    for xb, yb in batches(X, y, 1000, False):
        c += (m(xb).argmax(1) == yb).sum().item()
    return 100.0 * c / len(X)


def train(m, data, epochs, lr=1e-3, bs=128):
    Xtr, ytr, _, _ = data
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        for xb, yb in batches(Xtr, ytr, bs, True):
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep+1}/{epochs}  test acc {accuracy(m, data):.2f}%", flush=True)
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def size_kb(m, tag):
    p = MODELS / f"{tag}.pt"
    torch.save(m.state_dict(), p)
    return os.path.getsize(p) / 1024


def latency_ms(m, bs=1, n=200, threads=1):
    torch.set_num_threads(threads)
    m = copy.deepcopy(m).cpu().eval(); x = torch.randn(bs, 1, 28, 28)
    with torch.no_grad():
        for _ in range(30): m(x)          # warm-up
        ts = []
        for _ in range(n):
            t = time.perf_counter(); m(x); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)