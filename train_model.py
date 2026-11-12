
import pandas as pd
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load Preprocessed Data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

print("Dataset Loaded Successfully!")

# 2. Create Models
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

results = []

best_model = None
best_model_name = ""
best_f1 = -1

# 3. Train Models
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# 4. Model Comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 5. Best Model Evaluation
print("\nBest Model:", best_model_name)

best_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_predictions))

# 6. Save Model
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(best_model, "model/best_model.pkl")

joblib.dump(
    X_train.columns.tolist(),
    "model/feature_columns.pkl"
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
print("Best Model:", best_model_name)
print("Best Model Saved Successfully!")
