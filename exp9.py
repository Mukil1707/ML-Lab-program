# Compare Linear and Polynomial Regression

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Input Data
X = np.array([[1], [2], [3], [4], [5]])

# Output Data
Y = np.array([1, 4, 9, 16, 25])

# -----------------------------
# Linear Regression
# -----------------------------

linear_model = LinearRegression()

linear_model.fit(X, Y)

linear_prediction = linear_model.predict([[6]])

# -----------------------------
# Polynomial Regression
# -----------------------------

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()

poly_model.fit(X_poly, Y)

poly_prediction = poly_model.predict(poly.fit_transform([[6]]))

# -----------------------------
# Output
# -----------------------------

print("Linear Regression Prediction:")
print(linear_prediction[0])

print("\nPolynomial Regression Prediction:")
print(poly_prediction[0])
