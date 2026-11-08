LAB 05 – Frameworks and Automatic Differentiation

Student: Daisy Mwale (73403)

OBJECTIVE

The purpose of this laboratory was to investigate computation graphs,
automatic differentiation, execution modes, tensor representations and
framework overheads using PyTorch.

--------------------------------------------------

TASK 1: VALUE NODE DESIGN

A custom Value class was implemented to store:

- Data
- Gradient
- Parent nodes
- Local derivative rules

The Value class automatically constructs a computation graph during the
forward pass.

--------------------------------------------------

TASK 2: OPERATIONS

The following operations were implemented:

- Addition
- Multiplication
- Power
- ReLU

Each operation builds the computation graph automatically.

--------------------------------------------------

TASK 3: BACKWARD PROPAGATION

The backward pass was implemented using:

- Topological sorting
- Reverse graph traversal
- Gradient accumulation

Gradients are accumulated using += rather than overwritten.

--------------------------------------------------

TASK 4: GRADIENT VERIFICATION

Expression Tested:

((a * b + c) ** 2).relu()

Custom Engine:

30.0 -20.0 -10.0

PyTorch:

30.0 -20.0 -10.0

Result:

The custom automatic differentiation engine produced the same gradients
as PyTorch autograd.

Maximum absolute difference:

< 1e-6

--------------------------------------------------

TASK 5: MAKE_MOONS CLASSIFICATION

Dataset:

- make_moons
- 1000 samples
- Noise = 0.2

Network Architecture:

2 -> 8 -> 1

Activation Function:

Sigmoid

Loss Function:

Binary Cross Entropy (BCE)

Training Result:

Final Loss: 0.26736947894096375

The model successfully learned the classification task.

--------------------------------------------------

TASK 6: EXECUTION MODES AND TENSOR ANALYSIS

Execution Comparison:

Eager Time:
0.12645673751831055

Traced Time:
0.12397432327270508

Tensor Properties:

Shape:
torch.Size([1000, 1000])

Dtype:
torch.float32

Contiguous:
True

Transposed Shape:
torch.Size([1000, 1000])

Contiguous After Transpose:
False

Contiguity Speed Test:

Contiguous Time:
0.0

Non-Contiguous Time:
0.005103588104248047

Observation:

The transposed tensor became non-contiguous and showed slower execution
than the contiguous tensor.

--------------------------------------------------

WHEN COMPILATION HELPS

Compiled or traced execution can reduce framework overhead by converting
repeated operations into an optimized execution graph. Although an
initial setup cost exists, repeated inference and training workloads
often execute faster than standard eager execution. The performance
benefit becomes more noticeable for larger models and repeated runs.

--------------------------------------------------

CONCLUSION

A scalar automatic differentiation engine was successfully implemented
and validated against PyTorch autograd. A neural network was trained on
the make_moons dataset, execution modes were compared, and tensor
contiguity effects on performance were investigated.