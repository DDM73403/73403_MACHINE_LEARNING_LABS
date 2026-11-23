Lab 7: Compression - pruning and distillation

Name: Daisy Mwale
ID: 73403
Group: 3
Date: 17 November 2026

======================================================================
1. Baseline (15)
======================================================================

An oversized CNN was trained on Fashion-MNIST (60k train / 10k test): three 3x3 conv layers
(32, 64, 128 filters, ReLU + 2x2 max-pool), then FC 1152->768->10. Adam, lr 1e-3, batch 128,
12 epochs, seed 0. Latency is the median of 200 runs on a single CPU thread after warm-up.

| Metric | Value |
|---|---|
| Test accuracy | 91.95 % |
| Parameters | 985,866 |
| Size on disk (fp32 state_dict) | 3,855 KB |
| Latency, batch 1 | 0.891 ms |
| Latency, batch 128 | 71.4 ms |

About 90 % of the parameters sit in the first fully connected layer (885k of 986k), which
matters for where pruning has the most room.

======================================================================
2. Magnitude pruning (35)
======================================================================

Method. Weights of all conv and linear layers (biases excluded) were ranked by absolute value.
- Global: one threshold across all layers.
- Layer-wise: the same sparsity applied inside every layer (a separate threshold per layer).
- Structured: the lowest-L2-norm filters were removed from each conv layer (10 / 25 / 50 %), and a
  physically smaller network was built (the next layer's input channels and the FC input were sliced).

Each pruned model was fine-tuned for 3 epochs (Adam, lr 3e-4) with the mask re-applied after every
optimiser step, so pruned weights stay at zero. A 0 %-sparsity control received the same fine-tuning.

Reference. Fine-tuning the dense model for 3 more epochs raised accuracy from 91.95 % to 92.74 %
(the baseline was slightly under-trained). Pruned models are therefore compared with 92.74 %.

| Method | Weights removed | Acc before FT | Acc after FT | vs control (92.74) |
|---|---|---|---|---|
| Dense control | 0 % | 91.95 | 92.74 | - |
| Global | 25 % | 91.97 | 92.74 | 0.00 |
| Global | 50 % | 91.70 | 92.74 | 0.00 |
| Global | 75 % | 90.12 | 92.94 | +0.20 |
| Global | 90 % | 80.05 | 92.44 | -0.30 |
| Global | 95 % | 46.39 | 91.29 | -1.45 |
| Layer-wise | 25 % | 91.40 | 92.89 | +0.15 |
| Layer-wise | 50 % | 84.03 | 92.80 | +0.06 |
| Layer-wise | 75 % | 44.58 | 92.12 | -0.62 |
| Layer-wise | 90 % | 16.41 | 88.93 | -3.81 |
| Layer-wise | 95 % | 13.41 | 84.51 | -8.23 |
| Structured | 10.9 % (29/58/115 filters) | 91.38 | 92.60 | -0.14 |
| Structured | 26.6 % (24/48/96) | 85.26 | 92.30 | -0.44 |
| Structured | 51.9 % (16/32/64) | 63.86 | 92.11 | -0.63 |

Figure 1: Accuracy vs sparsity (outputs/figures/sparsity_curves.png)

Discussion.
- Global vs layer-wise. Up to 75 % the two behave the same (differences below the ~0.3 pt noise
  level of a single seed). Beyond that, global pruning is much more robust: at 95 % it keeps 91.29 %,
  while layer-wise drops to 84.51 %. The per-layer sparsities explain why. At 95 % global sparsity the
  network removes 95.9 % of fc1 (885k weights, the most redundant layer) but only 31.6 % of conv1
  (288 weights, no redundancy). Layer-wise pruning forces 95 % on conv1 and fc2 as well, which destroys
  the few weights those layers have. Accuracy before fine-tuning shows the same effect (46 % vs 13 %).
- Fine-tuning matters. Accuracy right after pruning falls sharply at high sparsity (global 95 %:
  46 %; layer-wise 90 %: 16 %), and 3 epochs with a fixed mask recover most of it.
- Structured vs unstructured. Removing whole filters is a coarser constraint: before fine-tuning,
  50 % structured gives 63.9 % while 50 % global unstructured gives 91.7 %. After fine-tuning the
  gap shrinks to 0.6 pt (92.11 % vs 92.74 %). The achieved parameter reduction is slightly above the
  filter fraction (e.g. 51.9 % for 50 %) because the FC layer's input also shrinks.
- Caveats. Single seed; accuracy is measured on the test set (no separate validation split);
  sparsity counts weights only.

======================================================================
3. Latency and storage (15)
======================================================================

Setup. Latency is the median of 300 runs (batch 1) or 40 runs (batch 128), single CPU thread,
after warm-up, with no other heavy programs running. Sparse size uses a flat index (uint32) plus
a float32 value per non-zero weight; biases are stored dense.

3.1 Sparse kernel micro-benchmark (fc1, 885k weights)

| fc1 sparsity (actual) | batch | dense (ms) | CSR (ms) | CSR speedup |
|---|---|---|---|---|
| 0 % | 1 | 0.0475 | 0.9944 | 0.05x |
| 0 % | 128 | 2.1945 | 12.4934 | 0.18x |
| 26 % | 1 | 0.0465 | 0.6815 | 0.07x |
| 26 % | 128 | 2.1776 | 9.8459 | 0.22x |
| 52 % | 1 | 0.0454 | 0.4405 | 0.10x |
| 52 % | 128 | 2.1760 | 6.4251 | 0.34x |
| 77 % | 1 | 0.0460 | 0.2173 | 0.21x |
| 77 % | 128 | 2.1929 | 2.5476 | 0.86x |
| 91 % | 1 | 0.0458 | 0.0885 | 0.52x |
| 91 % | 128 | 2.2048 | 0.8433 | 2.61x |
| 96 % | 1 | 0.0515 | 0.0523 | 0.98x |
| 96 % | 128 | 2.1626 | 0.4258 | 5.08x |

Dense time is flat in sparsity: the dense kernel executes every multiply-add whether the weight
is zero or not. CSR is slower than dense at low sparsity because every non-zero needs an index
lookup and the memory access is irregular, so vectorisation and cache prefetching are lost.
It only breaks even around 80 % sparsity at batch 128 and around 96 % at batch 1, and only
gains a clear speedup (5x) at 96 % with batch 128.

This win is limited to one layer. Whole-model latency is also set by the conv layers, which stay
dense (unstructured sparse convolution has no comparable kernel in PyTorch), and by fixed overhead.

3.2 Whole-model size and latency

Dense reference = global_0 (same weights as the baseline, measured under the same conditions).
Batch-1 latency of the baseline alone was 0.778 ms vs 0.637 ms for global_0, so run-to-run noise
is about 10-20 %; differences below ~10 % are not meaningful.

| Model | Weights removed | Dense file (KB) | Sparse file (KB) | Effective MACs (M) | Lat b1 (ms) | Lat b128 (ms) |
|---|---|---|---|---|---|---|
| Dense reference | 0 % | 3854 | 7702 | 8.34 | 0.637 | 63.2 |
| Global | 25 % | 3854 | 5778 | 7.14 | 0.642 | 66.5 |
| Global | 50 % | 3854 | 3855 | 5.87 | 0.642 | 63.0 |
| Global | 75 % | 3854 | 1931 | 3.84 | 0.647 | 63.9 |
| Global | 90 % | 3854 | 777 | 2.06 | 0.658 | 64.4 |
| Global | 95 % | 3854 | 392 | 1.27 | 0.655 | 65.6 |
| Layer-wise | 95 % | 3854 | 392 | 0.42 | 0.596 | 58.4 |
| Structured | 10.9 % | 3437 | n/a | 6.92 | 0.591 | 59.9 |
| Structured | 26.6 % | 2833 | n/a | 4.90 | 0.483 | 41.6 |
| Structured | 51.9 % | 1855 | n/a | 2.37 | 0.334 | 25.5 |

Figure 2: Latency and size (outputs/figures/latency_size.png)

Storage. A dense file never shrinks, because zeros are stored as zeros. The sparse format
(uint32 index + float32 value = 8 bytes per non-zero vs 4 bytes per dense weight) is larger than
dense below 50 % sparsity, breaks even at exactly 50 % (3855 vs 3854 KB), and then shrinks the file:
2.0x at 75 %, 5.0x at 90 %, 9.8x at 95 %. Storage reduction is real, but only above 50 % sparsity.

Latency. Unstructured sparsity gave no speedup: batch-1 latency stays at 0.64-0.66 ms and batch-128
latency at 63-66 ms for every global level, although the effective MACs fall 6.6x (8.34M to 1.27M).
Layer-wise 95 % removes 20x of the MACs (0.42M) and is only ~7 % faster, within noise.
Structured pruning, which really shrinks the tensors, does speed up: 1.9x at batch 1 and 2.5x
at batch 128 for the 52 % model (0.637 to 0.334 ms; 63.2 to 25.5 ms), and the gain scales with the
filters removed (1.1x, 1.3x, 1.9x at batch 1). Its MAC reduction (3.5x) is larger than its
latency gain (1.9-2.5x), since fixed overhead (memory movement, pooling, ReLU, framework calls) does not shrink.

Why unstructured sparsity rarely speeds up dense CPU kernels.
1. Dense kernels (BLAS GEMM, convolution via im2col/oneDNN) run a fixed loop over every element.
   A zero costs the same as any other number, and the benchmark shows dense fc1 time constant
   (about 0.046 ms at batch 1, 2.2 ms at batch 128) from 0 % to 96 % sparsity.
2. Sparse kernels skip zeros but pay for it: index lookups, irregular memory access, lost
   vectorisation and poor cache use. CSR was 21x slower than dense at 0 % sparsity (batch 1), only
   matched it at ~96 % (batch 1) and ~80 % (batch 128), and reached 5x faster only at 96 % with batch 128.
3. The speedup is confined to one layer. fc1 is only ~7 % of batch-1 latency, because the conv
   layers dominate the compute and stay dense. So even 5x on fc1 changes the whole model by a few percent.
4. Real gains need either a structure the hardware can exploit (removed filters/channels, or
   N:M sparsity on supporting hardware) or a dedicated sparse accelerator. MAC counts are therefore
   an upper bound on benefit, not a latency measurement.

======================================================================
4. Distillation (30)
======================================================================

Setup. Teacher = the baseline CNN (985,866 params, 91.95 %). Student = same architecture with
8/16/32 filters and a 320-unit hidden layer (101,578 params, 9.7x fewer). Loss:
alpha*CE(student, labels) + (1-alpha)*T^2*KL(softmax(teacher/T) || softmax(student/T)), T = 4,
alpha = 0.5, using the standard Hinton direction (the lab manual writes the arguments in the other
order; REVERSE_KL in the code switches this). Teacher logits were computed once on the training
images. Both students use Adam, lr 1e-3, batch 128, 15 epochs. For each seed the plain and distilled
student share the same initialisation and batch order, so the comparison is paired. 3 seeds.

| Seed | Plain student | Distilled student | Gain |
|---|---|---|---|
| 0 | 90.96 | 90.73 | -0.23 |
| 1 | 90.33 | 90.65 | +0.32 |
| 2 | 90.94 | 90.79 | -0.15 |
| mean +- std | 90.74 +- 0.36 | 90.72 +- 0.07 | -0.02 +- 0.30 |

| Model (seed 0) | Test acc | Params | Size (KB) | Lat b1 (ms) | Lat b128 (ms) |
|---|---|---|---|---|---|
| Teacher | 91.95 | 985,866 | 3854 | 0.792 | 62.1 |
| Student, plain | 90.96 | 101,578 | 400 | 0.194 | 12.4 |
| Student, distilled | 90.73 | 101,578 | 400 | 0.205 | 12.3 |

Figure 3: Student training curves (outputs/figures/distillation.png)

Findings.
- Compression. The student is 9.6x smaller on disk, about 3.6-4.1x faster at batch 1 and
  5.0-5.3x faster at batch 128 (range over two separate timing runs), at a cost of about 1.2 points
  of accuracy (91.95 vs 90.74). Because it is a smaller dense network, the speedup is real, unlike
  unstructured pruning.
- Distillation vs plain student. No benefit was measured: the mean difference (-0.02 pt) is far
  smaller than the seed-to-seed spread (0.30 pt), and the sign changes between seeds. With 3 seeds the
  test cannot detect an effect of a few tenths of a point, so the honest conclusion is "no
  evidence of improvement", not "distillation does not work".
- Variance. The distilled student varied less across seeds (std 0.07 vs 0.36), which is consistent
  with soft targets acting as a regulariser, but n = 3 is too small to claim it.
- Latency noise. Plain and distilled students have identical architecture, yet measured 0.194 vs
  0.205 ms, a 5 % difference that is pure measurement noise.
- Possible reasons (hypotheses, not tested here). The teacher is only 1.2 pt better than the plain
  student, so there is little headroom to transfer. The teacher's logits were computed on its own
  training images, where it is very confident, which reduces the extra information in the soft labels.
  T and alpha were fixed by the lab and not tuned.
- Cost. Distillation needs a trained teacher (12 extra epochs) plus a teacher forward pass per
  training batch, so it costs more training compute than the plain student for no measured gain.

======================================================================
5. Summary and conclusion (20)
======================================================================

All models were measured in one session (single CPU thread, other programs closed). Accuracy is test
accuracy. Pruned models have 12 epochs of baseline training + 3 fine-tuning epochs, so "15"
epochs; the distilled student's 27 includes the 12 epochs spent training its teacher.
Student accuracy is the mean of 3 seeds; the others are single runs.

| Model | Acc % | Data (epochs / labels) | Model (params / size KB) | Compute (MMACs / lat b1 ms / lat b128 ms) | Energy proxy (uJ) |
|---|---|---|---|---|---|
| Dense baseline | 91.95 | 12 / hard | 985,866 / 3854 | 8.34 / 0.649 / 62.7 | 669 |
| Global 90 % | 92.44 | 15 / hard | 985,866 / 3854 (sparse file 777) | 2.06 / 0.656 / 64.6 | 73* |
| Layer-wise 90 % | 88.93 | 15 / hard | 985,866 / 3854 (sparse file 777) | 0.83 / 0.622 / 62.0 | 68* |
| Structured 50 % | 92.11 | 15 / hard | 474,122 / 1855 | 2.37 / 0.341 / 24.9 | 314 |
| Student, plain | 90.74 | 15 / hard | 101,578 / 400 | 0.60 / 0.182 / 11.9 | 68 |
| Student, distilled | 90.72 | 27 / hard + teacher | 101,578 / 400 | 0.60 / 0.189 / 12.0 | 68 |

* Energy proxy = MACs x 4.6 pJ + stored weights x 640 pJ (one DRAM read each, batch 1; Horowitz,
ISSCC 2014, 45 nm). It is an estimate, not a measurement. Starred rows assume ideal sparse hardware
(optimistic bound); their measured latency used dense kernels.
Pruned-model accuracies should be compared with the fine-tuned dense control (92.74 %, section 2),
not with 91.95 %.

Figure 4: Accuracy vs latency (outputs/figures/summary_tradeoff.png)

Reduction relative to the dense baseline

| Model | Size on disk | Effective MACs | Latency b1 / b128 | Energy proxy | Acc vs reference |
|---|---|---|---|---|---|
| Global 90 % | 1.0x (5.0x in sparse format) | 4.0x fewer | 1.0x / 1.0x (no gain) | 9.2x (ideal HW) | -0.30 (vs 92.74) |
| Layer-wise 90 % | 1.0x (5.0x sparse) | 10x fewer | 1.0x / 1.0x (no gain) | 9.8x (ideal HW) | -3.81 (vs 92.74) |
| Structured 50 % | 2.1x | 3.5x fewer | 1.9x / 2.5x faster | 2.1x | -0.63 (vs 92.74) |
| Student, plain | 9.6x | 13.9x fewer | 3.6x / 5.3x faster | 9.8x | -1.21 (vs 91.95) |
| Student, distilled | 9.6x | 13.9x fewer | 3.4x / 5.2x faster | 9.8x | -1.23 (vs 91.95) |

By efficiency pillar
- Data. No method reduced the data used: every model trains on the same 60k images. Compression
  added training work: +3 epochs for pruning, and for distillation a 12-epoch teacher plus a teacher
  forward pass per batch. Soft labels did not improve accuracy measurably (section 4).
- Model. Size depends on the format. Unstructured pruning leaves the dense file unchanged and
  only shrinks a sparse file, and only above 50 % sparsity (5x at 90 %). Structured pruning and a
  smaller architecture shrink the model in any format (2.1x and 9.6x).
- Compute. MAC counts and measured latency disagree for unstructured models: 4-10x fewer MACs, no
  measurable speedup. Only models with smaller dense tensors got faster (1.9-3.6x at batch 1).
- Energy. In this proxy, weight fetches dominate (630 of 669 uJ for the baseline), so energy
  follows the number of stored weights more than MACs. That is why the student and the 90 %
  unstructured models land at similar values (68-73 uJ), but only the student gets there without
  needing sparse hardware.

Conclusion.
1. Unstructured pruning compresses storage but did not speed up inference on this CPU. It tolerates
   very high sparsity (global 90 %: 92.44 %, within noise of the 92.74 % control) but only pays off with
   a sparse file format, a sparse-aware kernel, or accelerator support. Layer-wise pruning is a worse
   allocation than global at high sparsity (88.93 % vs 92.44 % at 90 %), because it forces the tiny
   conv1 and fc2 layers to the same sparsity as the redundant fc1.
2. Structured pruning gave the best accuracy/latency trade-off from the baseline: removing 50 % of the
   filters cost 0.63 pt and made inference 1.9x (batch 1) to 2.5x (batch 128) faster, with a
   2.1x smaller file in plain dense format.
3. A small student trained from scratch gave the largest real reduction (9.6x size, 3.6-5.3x faster)
   for about 1.2 pt of accuracy. Distillation did not improve on it in this experiment (mean gain
   -0.02 +- 0.30 pt over 3 seeds), so its extra cost (teacher training) was not justified here.
4. Overall: reductions in parameters or MACs are not reductions in latency. Compression should be
   judged with measurements on the target hardware, and the method must match what that hardware
   can exploit.

Limitations. Single seed for the pruning experiments (differences below ~0.3 pt are noise);
accuracy is reported on the test set, with no separate validation split; latency was measured on one
laptop CPU, single thread, with about 10-20 % run-to-run variation; energy is an analytical proxy;
the structured model (474k params) and the student (102k) are not the same size, so they are not a
like-for-like comparison of pruning versus a small network.

======================================================================
Reproducibility
======================================================================

- Hardware: YOUR CPU MODEL (CPU only), Windows
- Software: Python YOUR VERSION, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu
- Seeds: 0 (baseline), 1 (pruning fine-tuning), 0-2 (students)
- Run from the LAB07 folder, in this order:

    python src\step1_baseline.py
    python src\step2_pruning.py
    python src\step3_latency.py
    python src\step4_distillation.py
    python src\step5_summary.py

Outputs: outputs/results/*.csv, outputs/figures/*.png, outputs/models/*.pt