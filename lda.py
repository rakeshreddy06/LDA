import numpy as np
import matplotlib.pyplot as plt

# Our playground data: [number of kids, noise level]
playground_data = np.array([
  [2, 3],  # Swing area
  [4, 5],  # Slide area
  [1, 1],  # Sandbox
  [5, 4]   # Monkey bars
])

# Calculate the special directions (eigenvectors)
covariance_matrix = np.cov(playground_data.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Plot our playground data
plt.figure(figsize=(10, 8))
plt.scatter(playground_data[:, 0], playground_data[:, 1], 
          c=['red', 'blue', 'green', 'purple'], 
          label=['Swings', 'Slide', 'Sandbox', 'Monkey bars'])

# Plot our special arrows (eigenvectors)
for i in range(2):
  plt.arrow(np.mean(playground_data[:, 0]), np.mean(playground_data[:, 1]), 
            eigenvectors[0, i]*eigenvalues[i], eigenvectors[1, i]*eigenvalues[i], 
            head_width=0.1, head_length=0.2, fc='k', ec='k', 
            label=f'Special Direction {i+1}')

plt.xlabel('Number of Kids')
plt.ylabel('Noise Level')
plt.title('Playground Activity Map')
plt.legend()
plt.grid(True)
plt.show()

print("How important each direction is (Eigenvalues):")
print(eigenvalues)

# Created/Modified files during execution:
# None