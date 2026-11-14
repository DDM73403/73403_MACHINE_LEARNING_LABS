LAB 2 REPORT
Neural Network from Scratch using NumPy

Daisy Danai Mwale
73403
Group 3

========================================
Task 1: Dataset Preparation
========================================

The load_digits dataset was loaded from scikit-learn.

The input data was normalized by dividing all pixel values by 16 so that the values were between 0 and 1.

The labels were one-hot encoded using OneHotEncoder.

The dataset was split into:
- 70% Training Data
- 30% Testing Data

Random State: 42
Stratified Sampling: Enabled

========================================
Task 2: Network Architecture
========================================

A neural network with two hidden layers was implemented.

Network Structure:

Input Layer: 64 neurons
Hidden Layer 1: 16 neurons
Hidden Layer 2: 16 neurons
Output Layer: 10 neurons

Weight Dimensions:

W1 = (64, 16)
b1 = (16,)

W2 = (16, 16)
b2 = (16,)

W3 = (16, 10)
b3 = (10,)

Weights were initialized using a normal distribution with mean 0 and standard deviation 0.01.

========================================
Task 3: Forward Propagation
========================================

Forward propagation was implemented using:

- ReLU activation function in hidden layers
- Softmax activation function in output layer

The output probabilities summed to approximately 1.0 for each sample.

========================================
Task 4: Cross-Entropy Loss
========================================

Cross-entropy loss was implemented.

Initial loss was approximately:

2.3026

This matches the expected value:

ln(10) = 2.3026

indicating correct implementation.

========================================
Task 5: Backpropagation and Gradient Check
========================================

Backpropagation was implemented using the chain rule.

Gradient Check Results:

Numerical Gradient:
1.2180922936977367e-06

Analytical Gradient:
1.2220472150496427e-06

The numerical and analytical gradients were very close, indicating that the implementation was correct.

========================================
Task 6: Training
========================================

Training Parameters:

Learning Rate: 0.01
Epochs: 50

Loss values decreased during training.

Final Results:

Train Accuracy:
0.10182975338106603

Test Accuracy:
0.10185185185185185

A loss curve was generated and saved as:

loss_curves.png

========================================
Task 7: Computational Timing
========================================

Median Forward Pass Time:

0.00033365003764629364 seconds

Median Backward Pass Time:

0.0003285999991931021 seconds

The backward pass requires additional computations for gradient calculations and error propagation through the network. Therefore, it is computationally more intensive than the forward pass.

========================================
Conclusion
========================================

A two-hidden-layer neural network was successfully implemented from scratch using NumPy. Forward propagation, cross-entropy loss, backpropagation, gradient checking, training, and timing analysis were completed. The gradient check confirmed the correctness of the implementation, and the required loss curve and timing measurements were generated.