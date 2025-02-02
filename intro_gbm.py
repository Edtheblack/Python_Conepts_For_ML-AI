import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Classifier
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm_model.fit(X_train, y_train)

# Make predictions
predictions = gbm_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Gradient Boosting Model Accuracy:", accuracy)

# Output:
# Gradient Boosting Model Accuracy: 1.0 (This output might vary slightly depending on the model's parameters and the random state)
