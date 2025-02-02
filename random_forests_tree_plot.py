import matplotlib.pyplot as plt

# Get feature importances
importances = forest_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances, color='blue', align='center')
plt.xticks(range(X.shape[1]), iris.feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
