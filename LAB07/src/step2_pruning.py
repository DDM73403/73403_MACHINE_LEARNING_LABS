import copy
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from pruning import *

seed_all(0)
data = load_data()
base = CNN().to(DEV)
base.load_state_dict(torch.load(MODELS / "baseline.pt", map_location=DEV))
base_acc = accuracy(base, data)
print("baseline acc:", base_acc)


def save(m, tag):
    torch.save({"c": m.c, "h": m.h, "state": m.state_dict()}, MODELS / f"{tag}.pt")


rows = []

# ---- unstructured: global and layer-wise
for kind, mk in (("global", global_masks), ("layer", layer_masks)):
    for s in SPARSITIES:
        if s == 0 and kind == "layer":                 # identical to global 0 -> reuse
            r = dict(rows[0]); r["kind"] = "layer"; rows.append(r); continue
        print(f"[{kind} {s}]", flush=True)
        m = copy.deepcopy(base)
        masks = mk(m, s); apply_masks(m, masks)
        before = accuracy(m, data)
        seed_all(1)

        finetune(m, data, 3, masks=masks)
        rows.append(dict(kind=kind, target=s, sparsity=weight_sparsity(m) * 100,
                         acc_before=before, acc_after=accuracy(m, data),
                         layers=str(per_layer_sparsity(m))))
        save(m, f"{kind}_{s}")
        print("  ->", rows[-1], flush=True)

# ---- structured: remove whole filters
base_w = sum(mod.weight.numel() for _, mod in prunable(base))
for f in STRUCT_FRACS:
    print(f"[structured {f}]", flush=True)
    m = structured_prune(base, f)
    before = accuracy(m, data)
    seed_all(1)
    finetune(m, data, 3)
    red = 1 - sum(mod.weight.numel() for _, mod in prunable(m)) / base_w
    rows.append(dict(kind="structured", target=f, sparsity=red * 100,
                     acc_before=before, acc_after=accuracy(m, data), layers=str(m.c)))
    save(m, f"structured_{f}")
    print("  ->", rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "pruning.csv", index=False)
print(df[["kind", "target", "sparsity", "acc_before", "acc_after"]].to_string())

# ---- plot
plt.figure(figsize=(7.5, 4.8))
for kind, mk in (("global", "o"), ("layer", "s"), ("structured", "^")):
    d = df[df.kind == kind]
    line, = plt.plot(d.sparsity, d.acc_after, mk + "-", label=f"{kind} (after fine-tune)")
    plt.plot(d.sparsity, d.acc_before, mk + "--", alpha=0.45, color=line.get_color(), label=f"{kind} (before fine-tune)")
plt.axhline(base_acc, color="k", ls=":", label="dense baseline")

plt.xlabel("weights removed (%)"); plt.ylabel("test accuracy (%)")
plt.title("Accuracy vs sparsity"); plt.grid(alpha=0.3); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGURES / "sparsity_curves.png", dpi=150)
print("saved figure")