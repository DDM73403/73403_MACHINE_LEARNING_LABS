import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# =====================
# TASK 1 - DATASET
# =====================

digits = load_digits()

X = digits.data
y = digits.target

print(X.shape)
print(y.shape)

X = X / 16.0

print(X.min())
print(X.max())

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

print(y_onehot.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_onehot,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

# =====================
# TASK 2 - NETWORK
# =====================

W = 16

W1 = np.random.normal(0, 0.01, (64, W))
b1 = np.zeros(W)

W2 = np.random.normal(0, 0.01, (W, W))
b2 = np.zeros(W)

W3 = np.random.normal(0, 0.01, (W, 10))
b3 = np.zeros(10)

print(W1.shape)
print(b1.shape)

print(W2.shape)
print(b2.shape)

print(W3.shape)
print(b3.shape)


# =====================
# ACTIVATION FUNCTIONS
# =====================
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# =====================
# LOSS
# =====================

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


# =====================
# FORWARD PASS
# =====================

def forward(X):

    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    a2 = relu(z2)

    z3 = a2 @ W3 + b3
    output = softmax(z3)

    return z1, a1, z2, a2, z3, output


# =====================
# BACKPROPAGATION
# =====================

def backward(X, y, z1, a1, z2, a2, output):

    m = X.shape[0]

    dz3 = output - y
    dW3 = (a2.T @ dz3) / m
    db3 = np.sum(dz3, axis=0) / m

    da2 = dz3 @ W3.T
    dz2 = da2 * relu_derivative(z2)

    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0) / m

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)

    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0) / m

    return dW1, db1, dW2, db2, dW3, db3


# =====================
# ACCURACY
# =====================

def accuracy(X, y):

    _, _, _, _, _, output = forward(X)

    predictions = np.argmax(output, axis=1)
    true_labels = np.argmax(y, axis=1)

    return np.mean(predictions == true_labels)

learning_rate = 0.01
losses = []

for epoch in range(50):

    z1, a1, z2, a2, z3, output = forward(X_train)

    loss = cross_entropy(y_train, output)
    losses.append(loss)

    dW1, db1, dW2, db2, dW3, db3 = backward(
        X_train,
        y_train,
        z1,
        a1,
        z2,
        a2,
        output
    )

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    if epoch % 5 == 0:
        print("Epoch", epoch, "Loss:", loss)

train_acc = accuracy(X_train, y_train)
test_acc = accuracy(X_test, y_test)

epsilon = 1e-5

original = W3[0, 0]

W3[0, 0] = original + epsilon
_, _, _, _, _, out_plus = forward(X_train)
loss_plus = cross_entropy(y_train, out_plus)

W3[0, 0] = original - epsilon
_, _, _, _, _, out_minus = forward(X_train)
loss_minus = cross_entropy(y_train, out_minus)

W3[0, 0] = original

numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

print("Numerical Gradient:", numerical_grad)
print("Analytical Gradient:", dW3[0, 0])

forward_times = []

for _ in range(100):
    start = time.perf_counter()
    forward(X_train)
    end = time.perf_counter()

    forward_times.append(end - start)

print("Median Forward Time:", np.median(forward_times))

backward_times = []

for _ in range(100):

    z1, a1, z2, a2, z3, output = forward(X_train)

    start = time.perf_counter()

    backward(
        X_train,
        y_train,
        z1,
        a1,
        z2,
        a2,
        output
    )

    end = time.perf_counter()

    backward_times.append(end - start)

print("Median Backward Time:", np.median(backward_times))

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)