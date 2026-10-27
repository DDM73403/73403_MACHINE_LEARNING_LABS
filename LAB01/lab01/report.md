Lab 1 Report

Goal

The goal of this lab was to create a reproducible machine learning environment, train baseline machine learning models, and measure key system performance metrics including training time, inference latency, model size, and memory usage.



Method

The Breast Cancer Wisconsin dataset was loaded using scikit-learn.



A stratified train-test split was performed using:



Training set: 70%

Testing set: 30%

random\_state = 42

Two baseline models were trained:



LogisticRegression(max\_iter=1000, random\_state=42)

RandomForestClassifier(n\_estimators=100, random\_state=42)

Measurements were collected using repeated runs and median timing values.



Results

Accuracy

Model	Accuracy

LogisticRegression	0.9415

RandomForest	0.9357

Training Time

Model	Median Time (s)

LogisticRegression	0.299030

RandomForest	0.105685

Inference Latency

Model	Median Latency (ms)

LogisticRegression	0.050050

RandomForest	1.999250

Model Size

Model	Size (KB)

LogisticRegression	1.03

RandomForest	284.07

Memory Usage

Model	Memory (MB)

LogisticRegression	\~0

RandomForest	0.66

Deployment Budget Analysis

Budget Limits

Environment	Memory	Latency	Model Size

Cloud	≥ 1 GB	≤ 100 ms	≤ 500 MB

Edge	256–1024 MB	≤ 50 ms	≤ 50 MB

Mobile	64–256 MB	≤ 20 ms	≤ 10 MB

TinyML	≤ 256 KB	≤ 10 ms	≤ 100 KB

Logistic Regression

Latency: 0.050050 ms

Size: 1.03 KB

Memory usage: negligible

Suitable for:



Cloud

Edge

Mobile

TinyML

Random Forest

Latency: 1.999250 ms

Size: 284.07 KB

Memory usage: 0.66 MB

Suitable for:



Cloud

Edge

Mobile

Not suitable for TinyML because the model exceeds the 100 KB size requirement.



Conclusions

Logistic Regression achieved the highest accuracy of 0.9415 while maintaining a very small model size.



Random Forest trained faster but produced a significantly larger model than Logistic Regression.



Logistic Regression satisfies all deployment budget categories, including TinyML, while Random Forest is better suited for Cloud, Edge, and Mobile environments.

