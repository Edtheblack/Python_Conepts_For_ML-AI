import matplotlib.pyplot as plt

# Data
names = ['Alice', 'Bob', 'Charles', 'David', 'Edward']
ages = [25, 30, 35, 40, 45]

# Create a bar chart
plt.figure(figsize=(10, 5))
plt.bar(names, ages, color='blue')
plt.title('Age of Individuals')
plt.xlabel('Names')
plt.ylabel('Age')
plt.show()
