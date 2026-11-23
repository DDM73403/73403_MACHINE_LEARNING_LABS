import io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from pruning import prunable


def load_model(tag):
    ck = torch.load(MODELS / f"{tag}.pt", map_location="cpu")
    if "state" in ck:
        m = CNN(ck["c"], ck["h"]); m.load_state_dict(ck["state"])
    else:                                   # baseline.pt is a raw state_dict
        m = CNN(); m.load_state_dict(ck)
    return m


def dense_kb(m):
    buf = io.BytesIO(); torch.save(m.state_dict(), buf)
    return len(buf.getbuffer()) / 1024


def sparse_kb(m):
    """uint32 flat index + float32 value per non-zero weight; biases stored dense."""
    arrs = {}
    for n, mod in prunable(m):
        w = mod.weight.detach().numpy().ravel()
        idx = np.flatnonzero(w).astype(np.uint32)
        arrs[n + "_i"], arrs[n + "_v"] = idx, w[idx].astype(np.float32)
        arrs[n + "_b"] = mod.bias.detach().numpy()
    buf = io.BytesIO(); np.savez(buf, **arrs)

    return len(buf.getbuffer()) / 1024


def eff_macs(m):
    sp = {"conv1": 28 * 28, "conv2": 14 * 14, "conv3": 7 * 7, "fc1": 1, "fc2": 1}
    return sum((mod.weight != 0).sum().item() * sp[n] for n, mod in prunable(m))


def fc_bench(W, bs, n=300):
    """Dense vs CSR matmul on the fc1 weight, single thread."""
    torch.set_num_threads(1)
    x = torch.randn(W.shape[1], bs)

    def timeit(f):
        for _ in range(20): f()
        ts = []
        for _ in range(n):
            t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
        return float(np.median(ts) * 1e3)

    dense = timeit(lambda: W @ x)
    try:
        Ws = W.to_sparse_csr(); csr = timeit(lambda: Ws @ x)
    except Exception as e:
        print("CSR failed:", e); csr = float("nan")
    return dense, csr


pr = pd.read_csv(RESULTS / "pruning.csv")
models = [("baseline", "baseline", 0.0, 0.0)]
for _, r in pr.iterrows():
    if r.kind == "layer" and r.target == 0:

        continue                                            # reused global_0 result
    models.append((f"{r.kind}_{r.target:g}", r.kind, r.target, r.sparsity))

rows = []
for tag, kind, target, sp in models:
    print("measuring", tag, flush=True)
    m = load_model(tag)
    rows.append(dict(model=tag, kind=kind, target=target, sparsity=sp,
                     params=n_params(m), nonzero=sum((mod.weight != 0).sum().item() for _, mod in prunable(m)),
                     dense_kb=dense_kb(m), sparse_kb=sparse_kb(m), eff_macs=eff_macs(m),
                     lat_b1_ms=latency_ms(m, 1, n=300), lat_b128_ms=latency_ms(m, 128, n=40)))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "latency_size.csv", index=False)
print(df.drop(columns=["kind", "target"]).round(3).to_string())

# ---- sparse-kernel micro-benchmark on fc1 (global-pruned models)
bench = []
for s in [0, 0.25, 0.5, 0.75, 0.9, 0.95]:
    W = load_model(f"global_{s:g}").fc1.weight.detach()
    for bs in (1, 128):
        d, c = fc_bench(W, bs)
        bench.append(dict(sparsity=s, batch=bs, dense_ms=d, csr_ms=c, csr_speedup=d / c))
bd = pd.DataFrame(bench)
bd.to_csv(RESULTS / "fc1_sparse_bench.csv", index=False)
print(bd.round(4).to_string())

# ---- plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for kind, mk in (("global", "o-"), ("layer", "s-"), ("structured", "^-")):
    d = df[df.kind == kind]
    ax[0].plot(d.sparsity, d.lat_b1_ms, mk, label=kind)
ax[0].axhline(df[df.model == "baseline"].lat_b1_ms.iloc[0], color="k", ls=":", label="dense baseline")

ax[0].set(xlabel="weights removed (%)", ylabel="CPU latency, batch 1 (ms)", title="Latency vs sparsity")
ax[0].legend(); ax[0].grid(alpha=.3)
g = df[df.kind == "global"]
ax[1].plot(g.sparsity, g.dense_kb, "o-", label="dense format")
ax[1].plot(g.sparsity, g.sparse_kb, "s-", label="sparse format (index+value)")
ax[1].set(xlabel="weights removed (%)", ylabel="size on disk (KB)", title="Storage, global pruning")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIGURES / "latency_size.png", dpi=150)
print("saved figure")