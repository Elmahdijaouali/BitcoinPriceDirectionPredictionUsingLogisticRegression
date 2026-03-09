import kagglehub

# Download dataset
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load CSV file
dataset = pd.read_csv("/kaggle/input/bitcoin-historical-data/btcusd_1-min_data.csv")

# Convert Timestamp to datetime
dataset['Date'] = pd.to_datetime(dataset['Timestamp'], unit='s')

# Create Target: 1 if next row Close > current Close
dataset['Target'] = (dataset['Close'].shift(-1) > dataset['Close']).astype(int)

# Drop last row
dataset = dataset[:-1]

# Features
X = dataset[['Open', 'High', 'Low', 'Close', 'Volume']]

# Target
y = dataset['Target']

# Train-test split (time series style)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Trading signal
signal = ['BUY' if p > 0.6 else 'SELL' for p in y_prob]

dataset_test = X_test.copy()
dataset_test['Signal'] = signal
dataset_test['Actual'] = y_test.values

print(dataset_test.tail(10))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

print("AUC:", auc)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Bitcoin Logistic Regression")
plt.legend()
plt.show()