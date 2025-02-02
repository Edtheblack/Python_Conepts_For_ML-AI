###numpy operations example 1:-

import numpy as np

# Create a simple NumPy array
array = np.array([1, 2, 3, 4, 5])
print("NumPy Array:", array)

# Perform element-wise operations
squared = array ** 2
print("Squared Values:", squared)

# Calculate the mean
mean_value = np.mean(array)
print("Mean of the array:", mean_value)


###numpy operations example 2 creating dataframe:-


import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charles'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Accessing data by column
print("\nAges of the individuals:")
print(df['Age'])


###numpy operations example 3 creating dataframe and filtering:-

import pandas as pd

# Sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charles', 'David', 'Edward'],
    'Age': [25, 30, 35, 40, 45],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}
df = pd.DataFrame(data)

# Filter to find people over 30 years old
older_than_30 = df[df['Age'] > 30]
print("People older than 30:")
print(older_than_30)

# Group by City and get mean age
grouped = df.groupby('City')['Age'].mean()
print("\nAverage Age by City:")
print(grouped)
