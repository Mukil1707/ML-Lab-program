# Linear Regression Program

from sklearn.linear_model import LinearRegression

# Training Data
# Hours Studied

X = [
    [1],
    [2],
    [3],
    [4],
    [5]
]

# Marks Scored

Y = [20, 40, 60, 80, 100]

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, Y)

# Test Data
test = [[6]]

# Prediction
result = model.predict(test)

# Output
print("Hours Studied:", test[0][0])

print("Predicted Marks:", result[0])
