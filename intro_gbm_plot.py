import matplotlib.pyplot as plt

# Plot training development
plt.figure(figsize=(10, 5))
plt.plot(gbm_model.train_score_, label='Loss During Training')
plt.title('GBM Training Process')
plt.xlabel('Number of Iterations')
plt.ylabel('Deviance')
plt.legend()
plt.show()
