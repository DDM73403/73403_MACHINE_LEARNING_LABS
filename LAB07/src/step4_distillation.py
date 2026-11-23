import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from common import *

T, ALPHA = 4.0, 0.5
EPOCHS, LR, BS = 15, 1e-3, 128
SEEDS = [0, 1, 2]
REVERSE_KL = False     # manual writes KL(student||teacher); standard Hinton KD is KL(teacher||student)


def kd_loss(s, t, y):
    ce = F.cross_entropy(s, y)
    if REVERSE_KL:
        kl = (F.softmax(s / T, 1) * (F.log_softmax(s / T, 1) - F.log_softmax(t / T, 1))).sum(1).mean()
    else:
        kl = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean")
    return ALPHA * ce + (1 - ALPHA) * T * T * kl


def dense_kb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf)
    return len(buf.getbuffer()) / 1024


data = load_data()
Xtr, ytr, Xte, yte = data

teacher = CNN().to(DEV)

teacher.load_state_dict(torch.load(MODELS / "baseline.pt", map_location=DEV))
teacher.eval()
with torch.no_grad():
    t_logits = torch.cat([teacher(Xtr[i:i + 1000]) for i in range(0, len(Xtr), 1000)])
print("teacher acc:", accuracy(teacher, data))


def train_student(seed, use_teacher):
    seed_all(seed)                                   # same init + batch order for both variants
    m = CNN((8, 16, 32), 320).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    hist = []
    for ep in range(EPOCHS):
        m.train()
        idx = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(Xtr), BS):
            j = idx[i:i + BS]
            out = m(Xtr[j])
            loss = kd_loss(out, t_logits[j], ytr[j]) if use_teacher else F.cross_entropy(out, ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        hist.append(accuracy(m, data))
    tag = "distilled" if use_teacher else "plain"
    print(f"seed {seed} {tag}: final acc {hist[-1]:.2f}", flush=True)
    return m, hist


seed_rows, hists, keep = [], {}, {}
for sd in SEEDS:
    plain, hp = train_student(sd, False)
    kd, hk = train_student(sd, True)
    seed_rows.append(dict(seed=sd, plain=hp[-1], distilled=hk[-1]))
    hists[sd] = (hp, hk)

    if sd == SEEDS[0]:
        keep = {"student_plain": plain, "student_kd": kd}

sdf = pd.DataFrame(seed_rows)
sdf["gain"] = sdf.distilled - sdf.plain
sdf.to_csv(RESULTS / "distill_seeds.csv", index=False)
print(sdf.to_string())
print("mean:", sdf[["plain", "distilled", "gain"]].mean().round(2).to_dict())
print("std :", sdf[["plain", "distilled", "gain"]].std().round(2).to_dict())

# ---- size / latency table (seed 0 models + teacher, same conditions)
rows = []
for name, m in [("teacher", teacher), ("student_plain", keep["student_plain"]), ("student_kd", keep["student_kd"])]:
    torch.save({"c": m.c, "h": m.h, "state": m.state_dict()}, MODELS / f"{name}.pt")
    rows.append(dict(model=name, acc=accuracy(m, data), params=n_params(m), dense_kb=dense_kb(m),
                     lat_b1_ms=latency_ms(m, 1, n=300), lat_b128_ms=latency_ms(m, 128, n=40)))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "distillation.csv", index=False)
print(df.round(3).to_string())

# ---- curves (seed 0)
hp, hk = hists[SEEDS[0]]
plt.figure(figsize=(6.5, 4.3))
plt.plot(range(1, EPOCHS + 1), hp, "o-", label="student, no teacher")
plt.plot(range(1, EPOCHS + 1), hk, "s-", label="student, distilled (T=4, alpha=0.5)")
plt.axhline(accuracy(teacher, data), color="k", ls=":", label="teacher")
plt.xlabel("epoch"); plt.ylabel("test accuracy (%)"); plt.grid(alpha=.3); plt.legend()
plt.title("Student training, seed 0"); plt.tight_layout()
plt.savefig(FIGURES / "distillation.png", dpi=150)
print("saved figure")