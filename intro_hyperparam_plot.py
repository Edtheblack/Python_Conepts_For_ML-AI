import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Assuming 'grid_search' is your GridSearchCV object after fitting
# Convert the grid_search results into a Pandas DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Focus on the mean test scores and parameter values
pivot_table = results.pivot_table(values='mean_test_score',
                                  index='param_max_depth',
                                  columns='param_n_estimators',
                                  aggfunc=np.max)

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Grid Search Results')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Maximum Depth of Trees (max_depth)')
plt.show()
