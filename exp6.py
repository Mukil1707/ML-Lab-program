# Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Training Data
# [Height, Weight]

X = [
    [150, 50],
    [160, 60],
    [170, 70],
    [180, 90],
    [155, 55],
    [175, 80]
]

# Class Labels
Y = [
    'Underweight',
    'Normal',
    'Normal',
    'Overweight',
    'Underweight',
    'Overweight'
]

# Create Model
model = GaussianNB()

# Train Model
model.fit(X, Y)

# Test Data
X_test = [
    [165, 65],
    [178, 85],
    [152, 52]
]

# Actual Output
Y_test = [
    'Normal',
    'Overweight',
    'Underweight'
]

# Prediction
prediction = model.predict(X_test)

# Output
print("Predicted Output:")
print(prediction)

# Confusion Matrix
cm = confusion_matrix(Y_test, prediction)

print("\nConfusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(Y_test, prediction)

print("\nAccuracy:")
print(accuracy * 100, "%")
