import torch
import torch.nn as nn
import time

model = nn.Sequential(
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

x = torch.randn(1000, 100)

# Eager Execution
start = time.time()

for _ in range(500):
    model(x)

eager_time = time.time() - start

print("Eager Time:", eager_time)

# Tracing
traced_model = torch.jit.trace(model, x)

start = time.time()

for _ in range(500):
    traced_model(x)

traced_time = time.time() - start

print("Traced Time:", traced_time)

# Tensor Information
print("\nTensor Properties")

tensor = torch.randn(1000, 1000)

print("Shape:", tensor.shape)
print("Dtype:", tensor.dtype)
print("Contiguous:", tensor.is_contiguous())

transposed = tensor.t()

print("Transposed Shape:", transposed.shape)
print("Contiguous After Transpose:", transposed.is_contiguous())

print("\nContiguity Speed Test")

start = time.time()
for _ in range(100):
    tensor.sum()
contiguous_time = time.time() - start

start = time.time()
for _ in range(100):
    transposed.sum()
noncontiguous_time = time.time() - start

print("Contiguous Time:", contiguous_time)
print("Non-Contiguous Time:", noncontiguous_time)