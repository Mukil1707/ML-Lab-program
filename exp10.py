# Expectation Maximization Algorithm

from sklearn.mixture import GaussianMixture
import numpy as np

# Sample Data
X = np.array([
    [1],
    [2],
    [3],
    [10],
    [11],
    [12]
])

# Create EM Model
model = GaussianMixture(n_components=2)

# Train Model
model.fit(X)

# Predict Clusters
result = model.predict(X)

# Output
print("Data Points:")
print(X)

print("\nCluster Assigned:")
print(result)
