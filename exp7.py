# Logistic Regression Program

from sklearn.linear_model import LogisticRegression

# Training Data
# [Hours Studied]

X = [
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
]

# Output
# 0 = Fail
# 1 = Pass

Y = [0, 0, 0, 1, 1, 1]

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X, Y)

# Test Data
test = [[3.5]]

# Prediction
result = model.predict(test)

# Output
print("Test Data:", test)

if result[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")
