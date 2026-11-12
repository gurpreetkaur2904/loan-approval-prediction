
import pandas as pd

# Load Cleaned Dataset
data = pd.read_csv("data/cleaned_loan_data.csv")

# 1. Check Dataset
print("Original Dataset Shape:", data.shape)

# 2. Separate Numerical and Categorical Columns
numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# 3. Check Unique Values
print("\nCategorical Unique Values:")
for col in categorical_cols:
    print(col, ":", data[col].unique())

# 4. Save Feature Engineering Dataset
data.to_csv("data/feature_engineered_loan_data.csv", index=False)

print("\nFeature Engineering Preparation Completed!")
