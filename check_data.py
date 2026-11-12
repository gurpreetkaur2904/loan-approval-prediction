
import pandas as pd
import os

# Check dataset path
file_path = "data/loan_data.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset file not found!")
else:
    # Load dataset
    data = pd.read_csv(file_path)

    print("\n========== LOAN DATASET REPORT ==========")

    print("\n1. First 5 Records:")
    print(data.head())

    print("\n2. Dataset Shape:")
    print(data.shape)

    print("\n3. Column Names:")
    print(data.columns.tolist())

    print("\n4. Dataset Information:")
    data.info()

    print("\n5. Missing Values:")
    print(data.isnull().sum())

    print("\n6. Duplicate Records:")
    print(data.duplicated().sum())

    print("\n7. Loan Status Distribution:")

    if "loan_status" in data.columns:
        print(data["loan_status"].value_counts())
    elif "Loan_Status" in data.columns:
        print(data["Loan_Status"].value_counts())
    else:
        print("Loan status column not found!")

    print("\n========== REPORT COMPLETED ==========")
