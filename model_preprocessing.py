
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load Feature Engineered Dataset
data = pd.read_csv("data/feature_engineered_loan_data.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# 2. Separate Features and Target
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# 3. Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

print("\nEncoding Completed!")

# 4. Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Display Dataset Shapes
print("\nTraining and Testing Data:")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 6. Save Processed Data
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print("All files saved inside data folder.")
