# Mobile Price Prediction

from sklearn.tree import DecisionTreeClassifier

# Training Data
# [RAM(GB), Storage(GB)]

X = [
    [2, 32],
    [3, 64],
    [4, 64],
    [6, 128],
    [8, 256]
]

# Price Category
# 0 = Low Price
# 1 = Medium Price
# 2 = High Price

Y = [0, 0, 1, 1, 2]

# Create Model
model = DecisionTreeClassifier()

# Train Model
model.fit(X, Y)

# Test Data
test = [[6, 128]]

# Prediction
result = model.predict(test)

# Output
print("Mobile Details:", test)

print("\nPredicted Price Category:")

if result[0] == 0:
    print("Low Price Mobile")

elif result[0] == 1:
    print("Medium Price Mobile")

else:
    print("High Price Mobile")
