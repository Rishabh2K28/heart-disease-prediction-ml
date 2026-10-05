# ❤️ Heart Disease Prediction using Machine Learning

## Overview

This project implements a machine learning pipeline for predicting the presence of heart disease from patient clinical and demographic attributes.

The project covers the complete machine learning workflow, including data loading, exploratory data analysis, data preprocessing, feature-target separation, model training, prediction and performance evaluation.

## Problem Statement

Heart disease prediction can be formulated as a binary classification problem where the model predicts whether a patient is likely to have heart disease based on clinical measurements.

### Target

* `0` → No heart disease
* `1` → Heart disease present

## Dataset

The project uses a heart disease dataset containing clinical attributes such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Slope
* Number of major vessels
* Thalassemia

Dataset source and attribution are provided in the notebook.

## Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## Model

The project uses Logistic Regression for binary classification.

The model is trained on the training dataset and evaluated on unseen test data.

## Evaluation

Model performance is evaluated using metrics such as:

* Accuracy Score
* Confusion Matrix
* Classification Report

The exact results produced by the notebook are documented in the corresponding notebook outputs.

## Exploratory Data Analysis

The notebook includes:

* Dataset inspection
* Statistical analysis
* Target distribution analysis
* Feature distribution visualization
* Correlation analysis
* Heatmap visualization

## How to Run

### Option 1 — Google Colab

Open the notebook in Google Colab and run the cells sequentially.

### Option 2 — Local Environment

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
heart_disease_prediction.ipynb
```

## Disclaimer

This project is intended for educational and machine learning experimentation purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Comparison of multiple classification algorithms
* Model serialization
* Interactive Streamlit interface
* Explainable AI using SHAP
* Deployment as a web application
