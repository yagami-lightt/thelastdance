import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("exp6.csv")

# DBSCAN for outlier detection
dbscan = DBSCAN(eps=0.5, min_samples=5)
df["cluster"] = dbscan.fit_predict(df[["X", "Y"]])

# Visualize DBSCAN clusters
outliers_dbscan = df[df["cluster"] == -1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["X"], y=df["Y"], hue=df["cluster"], palette="viridis", legend=False)
plt.scatter(outliers_dbscan["X"], outliers_dbscan["Y"], color="red", label="Outliers")
plt.title("DBSCAN Outlier Detection")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Labeling outliers
df["label"] = np.where(df["cluster"] == -1, 1, 0)

# Remove outliers for training model
df_cleaned = df[df["cluster"] != -1].drop(columns=["cluster"])
df_cleaned["label"] = 0

# Train and evaluate model before outlier removal
X_train, X_test, y_train, y_test = train_test_split(df[["X", "Y"]], df["label"], test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model performance before outlier removal
accuracy_before = accuracy_score(y_test, y_pred)
precision_before = precision_score(y_test, y_pred, zero_division=1)
recall_before = recall_score(y_test, y_pred, zero_division=1)
f1_before = f1_score(y_test, y_pred, zero_division=1)

# Train and evaluate model after outlier removal
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(
    df_cleaned[["X", "Y"]], df_cleaned["label"], test_size=0.2, random_state=42
)
model_cleaned = RandomForestClassifier(n_estimators=100, random_state=42)
model_cleaned.fit(X_train_cleaned, y_train_cleaned)
y_pred_cleaned = model_cleaned.predict(X_test_cleaned)

# Model performance after outlier removal
accuracy_cleaned = accuracy_score(y_test_cleaned, y_pred_cleaned)
precision_cleaned = precision_score(y_test_cleaned, y_pred_cleaned, zero_division=1)
recall_cleaned = recall_score(y_test_cleaned, y_pred_cleaned, zero_division=1)
f1_cleaned = f1_score(y_test_cleaned, y_pred_cleaned, zero_division=1)

# Compare performance before and after outlier removal
performance_comparison = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Before Outlier Removal": [accuracy_before, precision_before, recall_before, f1_before],
    "After Outlier Removal": [accuracy_cleaned, precision_cleaned, recall_cleaned, f1_cleaned],
}

performance_df = pd.DataFrame(performance_comparison)
print(performance_df)
