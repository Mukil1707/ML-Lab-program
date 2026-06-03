# House Price Prediction using Linear Regression

from sklearn.linear_model import LinearRegression

# Training Data
# [House Size in sqft]

X = [
    [1000],
    [1500],
    [2000],
    [2500],
    [3000]
]

# House Prices

Y = [2000000, 3000000, 4000000, 5000000, 6000000]

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, Y)

# Test Data
test = [[2200]]

# Prediction
price = model.predict(test)

# Output
print("House Size:", test[0][0], "sqft")

print("\nPredicted House Price:")
print(price[0])
