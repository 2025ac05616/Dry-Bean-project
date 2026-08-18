# Dry Bean Classification - Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to classify dry bean varieties using geometric and shape-related measurements extracted from bean images. Multiple machine learning classification models are trained and evaluated on the same dataset, and their performance is compared using standard classification metrics.

## 2. Dataset Description

The Dry Bean Dataset contains 13,611 instances and 16 numerical features describing the shape and geometric characteristics of dry beans.

The target variable is `Class`, containing 7 dry bean varieties:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

The 16 input features are:

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

The dataset was divided using a stratified 80/20 train-test split with `random_state=42`. The resulting test dataset contains 2,709 samples with all 7 classes represented.

## 3. GitHub Repository

https://github.com/2025ac05616/Dry-Bean-project

## 4. Models Used

The following classification models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.8970 | 0.9899 | 0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Random Forest | 0.9199 | 0.9907 | 0.9199 | 0.9199 | 0.9198 | 0.9031 |

## 5. Observations on Model Performance

### Logistic Regression

Logistic Regression achieved 91.92% accuracy and an AUC of 0.9934. It performed strongly across all evaluation metrics and provided excellent class separation.

### Decision Tree

Decision Tree achieved 89.66% accuracy. Its performance was lower than the other models, with an AUC of 0.9363 and MCC of 0.8750.

### K-Nearest Neighbor

KNN achieved 91.55% accuracy and an AUC of 0.9811. It performed competitively with Logistic Regression and Random Forest.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved 89.70% accuracy. Although its AUC was high at 0.9899, its accuracy and MCC were lower than Logistic Regression, KNN, and Random Forest.

### Random Forest

Random Forest achieved the highest accuracy of 91.99%, with an AUC of 0.9907 and MCC of 0.9031. It provided the strongest overall classification performance among the evaluated models.

### Overall Winner

Random Forest is the overall winner for this dataset based on its highest accuracy, precision, recall, F1 score, and MCC.

## 6. Streamlit Application

The project includes an interactive Streamlit application with the following features:

- CSV test-data upload
- Machine learning model selection
- Prediction results
- Prediction probability
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Confusion matrix
- Classification report
- Model comparison

## 7. Project Structure

```text
Dry-Bean-project/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── ML_Assignment_2.ipynb
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── feature_names.pkl
