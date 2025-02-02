import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
forest_model = RandomForestClassifier(n_estimators=100, random_state=42) # 100 trees in the forest
forest_model.fit(X_train, y_train)

# Make predictions
predictions = forest_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Random Forest Model Accuracy:", accuracy)

# Output:
# Random Forest Model Accuracy: 1.0 (This output might vary slightly depending on the random state and data split)
