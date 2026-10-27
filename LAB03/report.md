# Lab 3: Architectures and Their Costs
##The two graphs and tables are in the image.png 
![alt text](image.png)


## Results

| Architecture | Parameters | FLOPs | Latency (ms) | Accuracy (%) |
|-------------|-----------:|-----------:|-----------:|-----------:|
| MLP | 235146 | 234752 | 0.1404 | 88.08 |
| CNN | 421642 | 4241152 | 0.4425 | 90.87 |
| Attention | 101770 | 101632 | 0.7623 | 83.89 |

## Task 5: Architectural Trade-Off Analysis

The MLP achieved a test accuracy of 88.08% with 235,146 parameters and 234,752 FLOPs. Although it produced good classification results, it does not exploit spatial information in images because all pixels are treated independently through fully connected layers.

The CNN achieved the highest test accuracy of 90.87%. It required the largest number of parameters (421,642) and the highest computational cost (4,241,152 FLOPs). However, convolutional layers are able to extract meaningful image features and spatial patterns, making CNN the most effective architecture for Fashion-MNIST classification.

The Attention model achieved a test accuracy of 83.89%. It used the fewest parameters (101,770) and the lowest FLOPs (101,632), making it the most computationally efficient architecture. However, its classification accuracy was lower than both the MLP and CNN in this experiment.

Overall, the CNN provided the best trade-off between computational cost and predictive performance. Although the CNN required more computational resources, the increase in accuracy justified the additional cost, making it the most cost-effective architecture for the Fashion-MNIST dataset.

## Conclusion

Three neural network architectures were evaluated using the Fashion-MNIST dataset: MLP, CNN, and an Attention-based model. The CNN achieved the highest accuracy of 90.87% and demonstrated the best overall performance. The Attention model was the most computationally efficient due to its low parameter count and FLOPs but achieved the lowest accuracy. The MLP served as a strong baseline model with moderate computational requirements