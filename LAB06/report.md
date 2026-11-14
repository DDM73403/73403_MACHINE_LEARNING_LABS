Lab 6 – Training Pipelines and Project Proposal
Student: Daisy Mwale

Student ID: 73403

Task 1 – YAML-Driven Training Pipeline
A reproducible training pipeline was developed using a YAML configuration file.

Features implemented:

YAML-based configuration
FashionMNIST dataset loading
Reproducibility using fixed seed
MLflow experiment tracking
Checkpoint saving
Metric logging
Parameter logging
Results:

Accuracy: 83.40%
Checkpoint:

checkpoints/model.pth

Task 2 – Profiling
PyTorch profiler was used to profile one training iteration.

Component	Time (%)
Data Loading	84.34
Optimizer Step	7.11
Tensor Operations / Copies	9.88
Observation:

Data loading was identified as the primary bottleneck, accounting for approximately 84% of execution time.

Task 3 – num_workers Study
num_workers	Time (s)
0	6.90
1	11.54
2	8.89
4	7.69
8	8.72
Observation:

Increasing worker count did not improve performance on the test machine.

Best result:

num_workers = 0
Time = 6.90 s
Conclusion:

For the FashionMNIST dataset on the current hardware, multiprocessing overhead outweighed the benefits of additional workers

Task 4 – Batch Size Study
Batch Size	Throughput (samples/s)	Accuracy (%)
32	3699.92	80.93
64	4038.40	83.13
128	4532.24	83.07
256	4668.73	83.81
512	4767.38	84.01
Observations
Throughput increased as batch size increased.
Accuracy also improved slightly with larger batch sizes.
Batch size 512 achieved the highest throughput and accuracy.
Larger batch sizes process more samples per iteration, improving hardware utilization.
Conclusion
The best overall configuration in this study was batch size 512, achieving:

Throughput: 4767.38 samples/s
Accuracy: 84.01%

Task 5 – Optimizer and Scheduler Comparison
Optimizer	Scheduler	Accuracy (%)
Adam	StepLR	84.64
Adam	CosineAnnealingLR	84.57
Adam	ReduceLROnPlateau	83.93
SGD + Momentum	StepLR	84.61
SGD + Momentum	CosineAnnealingLR	84.61
SGD + Momentum	ReduceLROnPlateau	83.90
Observation:

Adam with StepLR produced the highest accuracy.

Best result:

Adam + StepLR
Accuracy = 84.64%


Task 6 – Gradient Accumulation
Configuration:

Physical batch size = 64
Accumulation steps = 4
Effective batch size = 256
Results:

Method	Accuracy (%)
Batch Size 256	83.81
Batch Size 64 + Accumulation 4	83.13
Observation:

The difference between the two training strategies was only 0.68%.

Conclusion:

Gradient accumulation successfully emulated a larger batch size while requiring less memory.

Task 7 – Checkpoint Recovery
A saved checkpoint was loaded successfully.

Output:

Checkpoint loaded successfully

Training resumed from saved model

Observation:

Training state was recovered successfully without restarting the experiment.



