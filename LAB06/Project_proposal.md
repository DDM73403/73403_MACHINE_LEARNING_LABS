# Individual Project Proposal

## Student Information

Name: Daisy Mwale

Student ID: 73403

## Project Title

Plant Disease Detection Using Deep Learning for Mobile Devices

## Background

Plant diseases are a major cause of crop loss worldwide. Early detection of plant diseases can help farmers take preventive measures and improve agricultural productivity. Traditional disease diagnosis often requires expert knowledge and manual inspection, which may not be readily available to all farmers.

## Problem Statement

The objective of this project is to develop a machine learning model capable of identifying plant diseases from images of plant leaves. The system will automatically classify whether a plant is healthy or affected by a specific disease, enabling faster and more accessible disease diagnosis.

## Dataset

The project will use the PlantVillage dataset, which contains thousands of labeled images of healthy and diseased plant leaves.

The dataset includes:

- Healthy leaf images
- Diseased leaf images
- Multiple plant species
- Multiple disease categories

## Proposed Methodology

A Convolutional Neural Network (CNN) will be trained to classify plant leaf images into disease categories. The training process will include data preprocessing, model training, evaluation, and optimization for deployment.

## Success Metrics

The model will be evaluated using:

- Classification Accuracy
- Precision
- Recall
- F1-Score

The solution will be considered successful if it achieves high accuracy on unseen test images while maintaining efficient performance.

## Deployment Constraints

The final model must satisfy the following requirements:

- Inference latency ≤ 50 ms
- Model size ≤ 5 MB

These constraints support deployment on mobile phones and edge devices commonly used in agricultural environments.

## Planned Optimization

Quantization will be applied to reduce model size and improve inference speed while preserving predictive performance.

## Expected Outcome

The expected outcome is a lightweight and accurate plant disease detection system that can assist farmers in identifying crop diseases quickly and efficiently. The model should meet deployment constraints while maintaining strong classification performance.

## Conclusion

This project applies deep learning techniques to agricultural image classification. By automating plant disease detection, the system has the potential to improve crop management, reduce losses, and demonstrate the practical deployment of machine learning solutions on resource-constrained devices.