import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC  # "Support vector classifier"
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Default values
svm_model.fit(X_train, y_train)

# Make predictions
predictions = svm_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("SVM Model Accuracy:", accuracy)

# Example output:
# SVM Model Accuracy: 0.97
