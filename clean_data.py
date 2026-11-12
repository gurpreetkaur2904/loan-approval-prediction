
import pandas as pd
from pathlib import Path

# Load original dataset
file_path = Path("data/loan_data.csv")
data = pd.read_csv(file_path)

print("Original Dataset Shape:", data.shape)

# 1. Clean column names
data.columns = data.columns.str.strip().str.lower()

# 2. Remove duplicate records
data.drop_duplicates(inplace=True)

# 3. Handle unwanted values
data.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
data.replace('?', pd.NA, inplace=True)

# 4. Convert numerical columns
numeric_columns = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score'
]

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 5. Fill missing numerical values
for col in numeric_columns:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# 6. Fill missing categorical values
categorical_columns = data.select_dtypes(
    include=['object']
).columns

for col in categorical_columns:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].mode()[0])

# 7. Remove extra spaces from text columns
for col in categorical_columns:
    data[col] = data[col].astype(str).str.strip()

# 8. Save cleaned dataset
output_path = Path("data/cleaned_loan_data.csv")
data.to_csv(output_path, index=False)

# 9. Display cleaning report
print("\n========== DATA CLEANING REPORT ==========")

print("Cleaned Dataset Shape:", data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:", data.duplicated().sum())

print("\nTarget Distribution:")
if 'loan_status' in data.columns:
    print(data['loan_status'].value_counts())

print("\nCleaned dataset saved successfully!")
print("File Location:", output_path)
