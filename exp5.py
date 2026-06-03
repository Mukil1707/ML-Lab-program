# Simple K-Nearest Neighbours (K-NN)

import math

# Training Data
# [Height, Weight, Class]

data = [
    [150, 50, 'Underweight'],
    [160, 60, 'Normal'],
    [170, 70, 'Normal'],
    [180, 90, 'Overweight']
]

# New data to classify
new_point = [165, 65]

# Value of K
k = 3

# Calculate distance
distances = []

for row in data:

    distance = math.sqrt(
        (row[0] - new_point[0])**2 +
        (row[1] - new_point[1])**2
    )

    distances.append([distance, row[2]])

# Sort distances
distances.sort()

# Find nearest neighbours
neighbors = distances[:k]

# Count classes
count = {}

for neighbor in neighbors:

    label = neighbor[1]

    if label in count:
        count[label] += 1
    else:
        count[label] = 1

# Find final class
result = max(count, key=count.get)

# Output
print("Nearest Neighbours:")
for n in neighbors:
    print(n)

print("\nNew Data:", new_point)

print("\nClassified as:")
print(result)
