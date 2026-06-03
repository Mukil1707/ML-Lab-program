# Car Price Prediction using Linear Regression

from sklearn.linear_model import LinearRegression

# Training Data
# [Car Age, Mileage]

X = [
    [1, 10000],
    [2, 20000],
    [3, 30000],
    [4, 40000],
    [5, 50000]
]

# Car Prices
Y = [900000, 800000, 700000, 600000, 500000]

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, Y)

# Test Data
# [Car Age, Mileage]

test = [[2, 25000]]

# Prediction
price = model.predict(test)

# Output
print("Car Details:", test)

print("\nPredicted Car Price:")
print(price[0])
