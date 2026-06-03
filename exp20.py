# Future Sales Prediction using Linear Regression

from sklearn.linear_model import LinearRegression

# Training Data
# Months

X = [
    [1],
    [2],
    [3],
    [4],
    [5]
]

# Sales Data

Y = [1000, 1500, 2000, 2500, 3000]

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, Y)

# Predict Future Sales for Month 6
test = [[6]]

# Prediction
future_sales = model.predict(test)

# Output
print("Month:", test[0][0])

print("\nPredicted Future Sales:")
print(future_sales[0])
