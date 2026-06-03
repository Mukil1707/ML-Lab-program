# Iris Flower Classification using KNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris Dataset
iris = load_iris()

# Input Data
X = iris.data

# Output Labels
Y = iris.target

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3
)

# Create KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Train Model
model.fit(X_train, Y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, prediction)

# Output
print("Predicted Output:")
print(prediction)

print("\nAccuracy:")
print(accuracy * 100, "%")
