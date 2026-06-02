# ID3 Algorithm Implementation

import math
from collections import Counter

# Training Dataset
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes']
]

# Attribute Names
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# Function to calculate entropy
def entropy(data):
    labels = [row[-1] for row in data]
    total = len(labels)
    counts = Counter(labels)

    ent = 0

    for count in counts.values():
        probability = count / total
        ent -= probability * math.log2(probability)

    return ent

# Function to calculate information gain
def information_gain(data, attr_index):

    total_entropy = entropy(data)

    values = set([row[attr_index] for row in data])

    weighted_entropy = 0

    for value in values:

        subset = [row for row in data if row[attr_index] == value]

        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

# Find best attribute
gains = []

for i in range(len(attributes)):
    gain = information_gain(data, i)
    gains.append(gain)

best_attribute = attributes[gains.index(max(gains))]

print("Information Gain of Attributes:\n")

for i in range(len(attributes)):
    print(attributes[i], ":", round(gains[i], 3))

print("\nBest Attribute Selected:")
print(best_attribute)

# New sample classification
new_sample = ['Sunny', 'Cool', 'High', 'Strong']

print("\nNew Sample:")
print(new_sample)

# Simple classification rule
if new_sample[0] == 'Overcast':
    result = 'Yes'

elif new_sample[0] == 'Sunny' and new_sample[2] == 'High':
    result = 'No'

else:
    result = 'Yes'

print("\nClassification Result:")
print(result)
