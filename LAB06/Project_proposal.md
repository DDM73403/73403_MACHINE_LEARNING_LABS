# Individual Project Proposal

## Student Information

Name: Daisy Mwale

Student ID: 73403

## Project Title

Offline Maize Leaf Disease Detection for Smallholder Farmers in Zimbabwe Using Lightweight Deep Learning on Low-End Mobile Devices

## Background

Maize is the staple food crop in Zimbabwe and is grown by most smallholder farmers. Foliar diseases such as gray leaf spot, northern leaf blight and common rust can reduce yields significantly if not identified early. Smallholder farmers often have limited access to agricultural extension officers, and many work in areas with unreliable mobile data. Most existing plant disease apps depend on internet connectivity, are trained on laboratory images, and are not tailored to local conditions.

## Problem Statement

This project aims to develop a lightweight deep learning model that identifies maize leaf diseases from smartphone photos and runs fully offline on low-end Android devices. The project also investigates a known weakness of existing systems: models trained on clean laboratory images often perform poorly on real field photos. The project will measure this performance gap and test whether it can be reduced for Zimbabwean field conditions.

## Research Questions

1. How much does accuracy drop when a model trained on laboratory images (PlantVillage) is tested on field images of maize?
2. Can data augmentation, transfer learning and fine-tuning on locally collected images reduce this gap?
3. Can the resulting model meet mobile deployment constraints (≤ 50 ms inference, ≤ 5 MB) without a major loss in accuracy?

## Dataset

- PlantVillage (maize classes): healthy, gray leaf spot, common rust and northern leaf blight, used for initial training.
- Field images: maize leaf photos collected under real conditions (farms, agricultural colleges, or extension services in Zimbabwe), taken with ordinary smartphones. These will be used for fine-tuning and, importantly, for testing.
- Public field-image datasets (for example PlantDoc) may supplement the data if local collection is limited.

Images will be labeled by class, and where possible verified by an agricultural officer or lecturer.

## Proposed Methodology

1. Preprocess images (resize, normalize) and apply augmentation that simulates field conditions (varied lighting, cluttered backgrounds, blur, rotation).
2. Train a baseline CNN using transfer learning with lightweight architectures (MobileNetV3-Small, EfficientNet-Lite).
3. Evaluate the baseline on a held-out field-image test set to quantify the lab-to-field gap.
4. Fine-tune with field images and augmentation, then re-evaluate.
5. Compress the best model using post-training quantization (TensorFlow Lite) and test on a low-end Android device.

## Success Metrics

- Accuracy, Precision, Recall and F1-Score (per class and overall)
- Accuracy gap between laboratory test images and field test images, before and after fine-tuning
- Confusion matrix to identify which diseases are most often confused

The project is successful if fine-tuning measurably narrows the lab-to-field gap while the model meets the deployment constraints.

## Deployment Constraints

- Inference latency ≤ 50 ms
- Model size ≤ 5 MB
- Fully offline operation on low-end Android smartphones

These constraints reflect the realities of smallholder farming: affordable phones and little or no internet access.

## Planned Optimization

Post-training quantization (INT8) will be applied to reduce model size and speed up inference. Accuracy before and after quantization will be compared to confirm that performance is preserved.

## Expected Outcome

A compact, offline-capable maize disease classifier suited to Zimbabwean field conditions, together with an analysis of how training data source (laboratory vs field) affects real-world performance. The project will show whether a small amount of local data can substantially improve a model built from public datasets.

## Conclusion

This project applies deep learning to a real agricultural problem in Zimbabwe while addressing a practical limitation of current plant disease detection systems. By focusing on offline use, low-cost devices and field-image performance, it demonstrates both technical rigor and local relevance.