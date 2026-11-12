<div align="center">

# 🏦 Loan Approval Prediction System
### *An End-to-End Machine Learning Solution for Credit Risk Assessment*

![Python](https://img.shields.io/badge/Python-387399?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F79A3E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

</div>

##  Project Overview
The **Loan Approval Prediction System** is a machine learning workflow designed to automate and optimize the loan application evaluation process. By analyzing applicant profiles—including credit history, income, loan amount, and employment details—the predictive model classifies whether a loan application should be approved or rejected.

This project covers the full end-to-end data pipeline: from raw data checks and cleaning, to feature engineering, model training, evaluation, and interactive deployment.

---

##  Key Features & Workflow

*  **Data Cleaning & Preprocessing (`clean_data.py`, `check_data.py`)**: Handles missing values, treats outliers, and verifies data integrity.
*  **Exploratory Data Analysis (`eda.py`)**: Generates key visual insights into feature distributions and correlation with loan status.
*  **Feature Engineering (`feature_engineering.py`)**: Encodes categorical variables and creates meaningful features to improve predictive performance.
*  **Model Training & Evaluation (`train_model.py`, `model_evaluation.py`)**:
  * Trains predictive classification algorithms (e.g., Logistic Regression, Decision Trees, Random Forest).
  * Evaluates performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.
*  **Interactive Web App (`app.py`)**: Streamlit user interface allowing users to input applicant details in real-time to obtain immediate predictions.

---

## 📂 Project Structure

```text
├── results/                     # Output visualizations & metrics
│   ├── loan_intent.png
│   ├── loan_status.png
│   ├── model_accuracy.png
│   ├── model_comparison.csv
│   └── roc_curve.png
├── app.py                       # Main Streamlit application
├── check_data.py                # Data validation script
├── clean_data.py                # Data cleaning pipeline
├── eda.py                       # Exploratory Data Analysis module
├── feature_engineering.py       # Feature transformation & scaling
├── model_evaluation.py          # Model metrics & performance visualization
├── model_preprocessing.py       # Preprocessing pipelines for inference
├── train_model.py               # Model training script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
mai eh readme ch likhta