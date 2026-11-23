import io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from pruning import prunable

E_MAC, E_DRAM = 4.6, 640.0      # pJ per fp32 MAC, pJ per 32-bit DRAM read (Horowitz, ISSCC 2014, 45 nm)
SP = {"conv1": 28 * 28, "conv2": 14 * 14, "conv3": 7 * 7, "fc1": 1, "fc2": 1}


def load(tag):
    ck = torch.load(MODELS / f"{tag}.pt", map_location="cpu")
    if "state" in ck:
        m = CNN(ck["c"], ck["h"]); m.load_state_dict(ck["state"])
    else:
        m = CNN(); m.load_state_dict(ck)
    return m.eval()


def dense_kb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf)
    return len(buf.getbuffer()) / 1024


def sparse_kb(m):
    arrs = {}
    for n, mod in prunable(m):
        w = mod.weight.detach().numpy().ravel(); idx = np.flatnonzero(w).astype(np.uint32)
        arrs[n + "_i"], arrs[n + "_v"] = idx, w[idx].astype(np.float32)
        arrs[n + "_b"] = mod.bias.detach().numpy()

    buf = io.BytesIO(); np.savez(buf, **arrs)
    return len(buf.getbuffer()) / 1024


def eff_macs(m):
    return sum((mod.weight != 0).sum().item() * SP[n] for n, mod in prunable(m))


def nonzero(m):
    return sum((mod.weight != 0).sum().item() + mod.bias.numel() for _, mod in prunable(m))


pr = pd.read_csv(RESULTS / "pruning.csv")
acc_pr = {f"{r.kind}_{r.target:g}": r.acc_after for _, r in pr.iterrows()}
seeds = pd.read_csv(RESULTS / "distill_seeds.csv")
acc_base = pd.read_csv(RESULTS / "distillation.csv").set_index("model").loc["teacher", "acc"]

# (label, model tag, accuracy, epochs over the 60k training images, labels, unstructured?)
spec = [
    ("Dense baseline",    "baseline",       acc_base,                    12, "hard",            False),
    ("Global 90 %",       "global_0.9",     acc_pr["global_0.9"],        15, "hard",            True),
    ("Layer-wise 90 %",   "layer_0.9",      acc_pr["layer_0.9"],         15, "hard",            True),
    ("Structured 50 %",   "structured_0.5", acc_pr["structured_0.5"],    15, "hard",            False),
    ("Student, plain",    "student_plain",  seeds.plain.mean(),          15, "hard",            False),
    ("Student, distilled", "student_kd",    seeds.distilled.mean(),      27, "hard + teacher",  False),
]

latency_ms(load("baseline"), 1, n=50)              # warm-up run, discarded
rows = []
for label, tag, acc, ep, lab, unstr in spec:
    print("measuring", tag, flush=True)
    m = load(tag)

    mac, nz = eff_macs(m), nonzero(m)
    rows.append(dict(model=label, acc=acc, epochs=ep, labels=lab, params=n_params(m),
                     dense_kb=dense_kb(m), sparse_kb=sparse_kb(m) if unstr else None,
                     mmacs=mac / 1e6, lat_b1=latency_ms(m, 1, n=300), lat_b128=latency_ms(m, 128, n=40),
                     energy_uj=(mac * E_MAC + nz * E_DRAM) * 1e-6, unstr=unstr))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "summary.csv", index=False)

lines = ["| Model | Acc % | Data: epochs / labels | Model: params / size KB | Compute: MMACs / lat b1 / b128 (ms) | Energy proxy (uJ) |",
         "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    size = f"{r.dense_kb:.0f}" + (f" (sparse {r.sparse_kb:.0f})" if r.unstr else "")
    en = f"{r.energy_uj:.0f}" + ("*" if r.unstr else "")
    lines.append(f"| {r.model} | {r.acc:.2f} | {r.epochs} / {r.labels} | {r.params:,} / {size} | "
                 f"{r.mmacs:.2f} / {r.lat_b1:.3f} / {r.lat_b128:.1f} | {en} |")
lines += ["", "*Energy proxy = MACs x 4.6 pJ + stored weights x 640 pJ (one DRAM read each, batch 1). "
              "*Unstructured rows assume ideal sparse hardware, so they are an optimistic bound; "
              "measured latency for them used dense kernels."]
open(RESULTS / "summary_table.md", "w").write("\n".join(lines))
print("\n".join(lines))

plt.figure(figsize=(6.8, 4.5))
for _, r in df.iterrows():
    plt.scatter(r.lat_b1, r.acc, s=60)
    plt.annotate(r.model, (r.lat_b1, r.acc), fontsize=7, xytext=(4, 4), textcoords="offset points")
plt.xlabel("CPU latency, batch 1 (ms)"); plt.ylabel("test accuracy (%)")
plt.title("Accuracy vs measured latency"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIGURES / "summary_tradeoff.png", dpi=150)
