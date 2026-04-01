# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("churn.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())

# ----------------------------
# Data Cleaning
# ----------------------------
print("\nMissing Values:\n", data.isnull().sum())

# Drop ID column
if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

# Fix TotalCharges
if 'TotalCharges' in data.columns:
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

# ----------------------------
# Encode categorical data
# ----------------------------
le = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# ----------------------------
# EDA (Visualization)
# ----------------------------

# Churn Count
plt.figure()
sns.countplot(x='Churn', data=data)
plt.title("Churn Count (0 = No, 1 = Yes)")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# Split Data
# ----------------------------
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Model Training
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# ROC Curve
# ----------------------------
y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ----------------------------
# Feature Importance
# ----------------------------
importance = pd.Series(model.coef_[0], index=X.columns)

importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()