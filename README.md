# ❤️ Heart Disease Prediction using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishabh2K28/heart-disease-prediction-ml/blob/main/heart_disease_prediction.ipynb)

## Overview

This project implements an end-to-end machine learning pipeline for predicting the presence of heart disease from patient clinical and demographic data.

The project covers the complete workflow, including exploratory data analysis, data preprocessing, feature-target separation, train-test splitting, model training, prediction, and performance evaluation.

## Problem Statement

Heart disease prediction can be treated as a **binary classification problem**, where a machine learning model predicts whether a patient is likely to have heart disease based on clinical attributes.

### Target Variable

* `0` → No heart disease
* `1` → Heart disease present

## Dataset

The dataset contains **303 patient records and 14 attributes**, including the target variable.

### Features

* `age` — Age of the patient
* `sex` — Gender of the patient
* `cp` — Chest pain type
* `trestbps` — Resting blood pressure
* `chol` — Serum cholesterol
* `fbs` — Fasting blood sugar
* `restecg` — Resting electrocardiographic results
* `thalach` — Maximum heart rate achieved
* `exang` — Exercise-induced angina
* `oldpeak` — ST depression induced by exercise
* `slope` — Slope of the peak exercise ST segment
* `ca` — Number of major vessels
* `thal` — Thalassemia
* `target` — Heart disease outcome

The dataset used for the project is included in this repository as `data.csv`.

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
Logistic Regression Training
   ↓
Prediction
   ↓
Model Evaluation
```

## Exploratory Data Analysis

The notebook performs exploratory analysis to understand the dataset and relationships between the features and target variable.

The analysis includes:

* Dataset inspection
* Statistical analysis
* Target variable distribution
* Feature distributions
* Correlation analysis
* Correlation heatmap

## Model

The project uses **Logistic Regression** as the binary classification algorithm.

The dataset is divided into training and testing subsets, and the model is trained on the training data before being evaluated on unseen test data.

## Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

The corresponding results and evaluation outputs are available in the Jupyter Notebook.

## Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

## Project Structure

```text
heart-disease-prediction-ml/
│
├── data.csv
├── heart_disease_prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## How to Run

### Google Colab

Open `heart_disease_prediction.ipynb` in Google Colab and run the cells sequentially.

### Local Environment

Clone the repository:

```bash
git clone https://github.com/Rishabh2K28/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
heart_disease_prediction.ipynb
```

Make sure `data.csv` is available in the project directory.

## Repository

**GitHub:**
https://github.com/Rishabh2K28/heart-disease-prediction-ml

## Future Improvements

* Compare Logistic Regression with other classification algorithms
* Perform hyperparameter tuning and cross-validation
* Add additional evaluation metrics
* Build an interactive Streamlit interface
* Serialize and deploy the trained model
* Add model explainability using SHAP

## Disclaimer

This project is developed for educational and machine learning experimentation purposes only. It is **not a medical diagnostic system** and should not be used for clinical decision-making.

## Credits

This project was developed as a hands-on machine learning implementation based on the heart disease prediction tutorial by Siddhardhan.

The implementation has been organized into a reproducible GitHub repository along with the dataset, notebook, dependencies, and documentation.
