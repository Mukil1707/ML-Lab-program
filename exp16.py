# Compare Classification Algorithms

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load Dataset
iris = load_iris()

X = iris.data
Y = iris.target

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3
)

# -----------------------------
# KNN
# -----------------------------
knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(Y_test, knn_pred)

# -----------------------------
# Naive Bayes
# -----------------------------
nb = GaussianNB()

nb.fit(X_train, Y_train)

nb_pred = nb.predict(X_test)

nb_acc = accuracy_score(Y_test, nb_pred)

# -----------------------------
# Decision Tree
# -----------------------------
dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(Y_test, dt_pred)

# -----------------------------
# Logistic Regression
# -----------------------------
lr = LogisticRegression(max_iter=200)

lr.fit(X_train, Y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(Y_test, lr_pred)

# -----------------------------
# Output
# -----------------------------
print("Accuracy of KNN:")
print(knn_acc * 100, "%")

print("\nAccuracy of Naive Bayes:")
print(nb_acc * 100, "%")

print("\nAccuracy of Decision Tree:")
print(dt_acc * 100, "%")

print("\nAccuracy of Logistic Regression:")
print(lr_acc * 100, "%")
