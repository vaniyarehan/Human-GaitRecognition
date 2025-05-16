# Human Gait Recognition Using Transfer Learning and SVM

This project implements a human gait recognition pipeline based on the CASIA-B dataset. It leverages deep feature extraction from a pretrained ResNet101 model, applies feature selection and fusion techniques, and performs classification using a One-against-All Support Vector Machine (SVM).

## Features
- Subsets the CASIA-B dataset for 10 subjects with normal walking sequence.
- Extracts deep features using a pretrained ResNet101 network.
- Applies kurtosis-based feature selection.
- Fuses features based on correlation.
- Trains and evaluates an One-against-All SVM classifier.
- Provides classification report, test accuracy, and 5-fold cross-validation accuracy.

## Requirements
- Python
- numpy
- optionally matplotlib if you want to visualize the results
- scikit-learn
- scipy
- torch
- torchvision

You can install dependencies using:
```bash
pip install numpy matplotlib scikit-learn scipy torch torchvision pillow
