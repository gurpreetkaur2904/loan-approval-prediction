
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Professional graph style
sns.set_theme(style="whitegrid", palette="Set2")

# Load cleaned dataset
data = pd.read_csv("data/cleaned_loan_data.csv")

# Create output folder
os.makedirs("results", exist_ok=True)

print("========== EXPLORATORY DATA ANALYSIS ==========")

print("\nDataset Shape:", data.shape)

print("\nStatistical Summary:")
print(data.describe())

# 1. Loan Status Distribution
if "loan_status" in data.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="loan_status")
    plt.title("Loan Approval Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Number of Applicants")
    plt.tight_layout()
    plt.savefig("results/loan_status.png")
    plt.show()

# 2. Gender Distribution
if "person_gender" in data.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="person_gender")
    plt.title("Applicant Gender Distribution")
    plt.tight_layout()
    plt.savefig("results/gender.png")
    plt.show()

# 3. Education Distribution
if "person_education" in data.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="person_education")
    plt.title("Education Level Distribution")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig("results/education.png")
    plt.show()

# 4. Income Distribution
if "person_income" in data.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data["person_income"], bins=30, kde=True)
    plt.title("Applicant Income Distribution")
    plt.tight_layout()
    plt.savefig("results/income.png")
    plt.show()

# 5. Loan Amount Distribution
if "loan_amnt" in data.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data["loan_amnt"], bins=30, kde=True)
    plt.title("Loan Amount Distribution")
    plt.tight_layout()
    plt.savefig("results/loan_amount.png")
    plt.show()

# 6. Credit Score Analysis
if "credit_score" in data.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data["credit_score"], bins=30, kde=True)
    plt.title("Credit Score Distribution")
    plt.tight_layout()
    plt.savefig("results/credit_score.png")
    plt.show()

# 7. Loan Intent Analysis
if "loan_intent" in data.columns:
    plt.figure(figsize=(9, 5))
    sns.countplot(data=data, x="loan_intent")
    plt.title("Loan Purpose Distribution")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig("results/loan_intent.png")
    plt.show()

# 8. Correlation Heatmap
numeric_data = data.select_dtypes(include="number")

plt.figure(figsize=(11, 7))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("results/correlation.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("All graphs saved inside results folder.")
