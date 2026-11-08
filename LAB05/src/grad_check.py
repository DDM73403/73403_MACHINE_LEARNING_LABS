import torch
from value import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(1.0)

y = ((a * b + c) ** 2).relu()

y.backward()

print("Custom Engine")
print(a.grad, b.grad, c.grad)


a_t = torch.tensor(2.0, requires_grad=True)
b_t = torch.tensor(-3.0, requires_grad=True)
c_t = torch.tensor(1.0, requires_grad=True)

y_t = torch.relu((a_t * b_t + c_t) ** 2)

y_t.backward()

print("\nPyTorch")
print(
    a_t.grad.item(),
    b_t.grad.item(),
    c_t.grad.item()
)