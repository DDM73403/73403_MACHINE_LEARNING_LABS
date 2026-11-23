# Lab 7: Compression - pruning and distillation

**Name:** Daisy Mwale
**ID:** 73403
**Group:** 3

## What this lab does

Trains an oversized CNN (about 1M parameters) on Fashion-MNIST, compresses it with
magnitude pruning (global, layer-wise, structured) and knowledge distillation, and compares
the models on accuracy, size, latency, compute and energy.

## Folder structure

    LAB07/
      src/
        common.py              shared code: data, CNN, training, accuracy, latency
        pruning.py             masks, fine-tuning, structured filter pruning
        step1_baseline.py      train the 986k-parameter baseline
        step2_pruning.py       global / layer-wise / structured pruning + accuracy curves
        step3_latency.py       dense vs sparse size, latency, sparse-kernel benchmark
        step4_distillation.py  student with and without teacher (T=4, alpha=0.5, 3 seeds)
        step5_summary.py       final comparison table by efficiency pillar
      outputs/
        models/                saved model weights (.pt)
        results/               CSV result tables
        figures/               plots (.png)
      report.md                full report (report.txt is the same content as plain text)
      README.md
      requirements.txt

## Requirements


Python 3 with PyTorch, torchvision, NumPy, pandas and Matplotlib:

    pip install -r requirements.txt

Tested with PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu (CPU only).
Fashion-MNIST is downloaded automatically to `data/` on the first run (needs internet).

## How to run

Run from the `LAB07` folder, in this order (each step uses the output of the previous ones):

    python src\step1_baseline.py
    python src\step2_pruning.py
    python src\step3_latency.py
    python src\step4_distillation.py
    python src\step5_summary.py

Close other heavy programs before steps 3 to 5, since latency is sensitive to background load.

## Outputs

- `outputs/results/`: baseline.json, pruning.csv, latency_size.csv, fc1_sparse_bench.csv,
  distill_seeds.csv, distillation.csv, summary.csv, summary_table.md
- `outputs/figures/`: sparsity_curves.png, latency_size.png, distillation.png, summary_tradeoff.png
- `outputs/models/`: baseline.pt and all pruned and student models

## Main results

| Model | Test acc % | Params | Size KB | Latency b1 (ms) |
|---|---|---|---|---|
| Dense baseline | 91.95 | 985,866 | 3854 | 0.649 |

| Global pruning 90 % | 92.44 | 985,866 (777 KB sparse file) | 3854 | 0.656 (no speedup) |
| Structured pruning 50 % | 92.11 | 474,122 | 1855 | 0.341 |
| Student, plain | 90.74 | 101,578 | 400 | 0.182 |
| Student, distilled | 90.72 | 101,578 | 400 | 0.189 |

- Unstructured pruning shrinks the sparse file but gives no CPU speedup.
- Structured pruning and a smaller student give real speedups.
- Distillation showed no measurable gain over the plain student (mean difference -0.02 +- 0.30 pt, 3 seeds).

See `report.md` for the full analysis.

## Notes

- The distillation loss uses the standard KL(teacher || student). The lab manual writes the
  arguments the other way round; set `REVERSE_KL = True` in `step4_distillation.py` to follow it literally.
- The energy figures are an analytical proxy, not a measurement.