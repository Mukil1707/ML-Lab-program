# Credit Score Classification

from sklearn.tree import DecisionTreeClassifier

# Training Data
# [Income, Loan Amount]

X = [
    [25000, 5000],
    [30000, 7000],
    [40000, 10000],
    [60000, 15000],
    [80000, 20000]
]

# Credit Score Class
# 0 = Poor
# 1 = Good

Y = [0, 0, 1, 1, 1]

# Create Model
model = DecisionTreeClassifier()

# Train Model
model.fit(X, Y)

# Test Data
test = [[35000, 8000]]

# Prediction
result = model.predict(test)

# Output
print("Test Data:", test)

if result[0] == 1:
    print("Credit Score: Good")
else:
    print("Credit Score: Poor")
