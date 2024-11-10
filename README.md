# Chest X-ray Classification using Transfer Learning with MobileNetV2 and DenseNet169
This project implements a **stacked deep learning model** for classifying chest X-ray images, leveraging the pre-trained **MobileNetV2** and **DenseNet169** architectures. Using transfer learning, the model aims to detect abnormalities in X-ray images, providing a robust approach to assist in early diagnosis.

## Table of Contents
+ Project Overview
+ Dataset
+ Model Architecture
+ Usage
+ Results
+ Improvements

## Project Overview
Medical imaging, such as chest X-rays, is critical in diagnosing various conditions, including pneumonia and tuberculosis. The goal of this project is to classify X-ray images as Normal or Pneumonia using a combined transfer learning model that stacks **MobileNetV2** and **DenseNet169**.

By stacking these models, we aim to leverage the unique features captured by each and **improve classification accuracy**, especially for detecting subtle features in medical images.

## Dataset
The dataset consists of labeled chest X-ray images in two classes:

+ Normal: No abnormalities
+ Pneumonia: Presence of abnormalities (i.e, pneumonia)

## Model Architecture
+ **Transfer Learning Base Models:** MobileNetV2 and DenseNet169 are used as base models, both pre-trained on the ImageNet dataset.
+ **Stacked Architecture:** The outputs of both base models are combined to create a stronger feature representation.
+ **Custom Layers:** Fully connected (dense) layers and dropout layers are added after the concatenation of base model outputs.
+ **Output Layer:** A softmax layer for binary classification.

### Key Components:
+ Data Augmentation: Applied only to the training data to improve generalization.
+ Class Weights: Used to handle class imbalance in the dataset.
+ Batch Normalization and Dropout: Used to prevent overfitting.

## Usage
+ Preprocess the Data: Ensure your dataset is structured in directories for train, val, and (optionally) test.
+ Train the Model: Run the code provided in the notebook to train the stacked model.
+ Evaluate: The model evaluates performance on the validation set, which should be free of data augmentation for accurate benchmarking.

## Results
### Model Performance

| Metric               | Value |
|----------------------|---------------|
| Validation Loss      | Content Cell  |
| Validation Loss      | Content Cell  |

### Training/Validation Accuracy Curve

## Improvements
### Possible improvements to explore in future iterations:

+ Fine-Tuning: Unfreeze top layers of MobileNetV2 and DenseNet169 for fine-tuning.
+ Alternative Models: Try EfficientNet for potentially improved accuracy.
+ Advanced Augmentation: Experiment with Albumentations library for additional augmentations.
+ Ensembling: Use ensembling techniques with other models to enhance predictions.

## Contribution 
Feel free to make some changes
  
