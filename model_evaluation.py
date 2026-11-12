
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Create Results Folder
os.makedirs("results", exist_ok=True)

# 1. Load Saved Model and Testing Data
model = joblib.load("model/best_model.pkl")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

# 2. Make Predictions
y_pred = model.predict(X_test)

print("MODEL EVALUATION RESULTS")
print("------------------------")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Approved", "Approved"],
    yticklabels=["Not Approved", "Approved"]
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

# 4. Model Accuracy Comparison
results = pd.read_csv("results/model_comparison.csv")

plt.figure(figsize=(8, 5))
sns.barplot(
    data=results,
    x="Model",
    y="Accuracy",
    hue="Model",
    palette="viridis",
    legend=False
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/model_accuracy.png", dpi=300)
plt.show()

# 5. ROC Curve
y_probability = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probability)
roc_auc = roc_auc_score(y_test, y_probability)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=300)
plt.show()

# 6. Feature Importance
if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="magma",
        legend=False
    )

    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=300)
    plt.show()

print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("All graphs saved inside results folder.")
