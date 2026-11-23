| Model | Acc % | Data: epochs / labels | Model: params / size KB | Compute: MMACs / lat b1 / b128 (ms) | Energy proxy (uJ) |
|---|---|---|---|---|---|
| Dense baseline | 91.95 | 12 / hard | 985,866 / 3854 | 8.34 / 0.649 / 62.7 | 669 |
| Global 90 % | 92.44 | 15 / hard | 985,866 / 3854 (sparse 777) | 2.06 / 0.656 / 64.6 | 73* |
| Layer-wise 90 % | 88.93 | 15 / hard | 985,866 / 3854 (sparse 777) | 0.83 / 0.622 / 62.0 | 68* |
| Structured 50 % | 92.11 | 15 / hard | 474,122 / 1855 | 2.37 / 0.341 / 24.9 | 314 |
| Student, plain | 90.74 | 15 / hard | 101,578 / 400 | 0.60 / 0.182 / 11.9 | 68 |
| Student, distilled | 90.72 | 27 / hard + teacher | 101,578 / 400 | 0.60 / 0.189 / 12.0 | 68 |

*Energy proxy = MACs x 4.6 pJ + stored weights x 640 pJ (one DRAM read each, batch 1). *Unstructured rows assume ideal sparse hardware, so they are an optimistic bound; measured latency for them used dense kernels.