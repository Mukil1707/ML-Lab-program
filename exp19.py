# Bank Loan Prediction using Naive Bayes

from sklearn.naive_bayes import GaussianNB

# Training Data
# [Income, Credit Score]

X = [
    [25000, 600],
    [30000, 650],
    [40000, 700],
    [50000, 750],
    [60000, 800]
]

# Loan Approval
# 0 = No Loan
# 1 = Loan Approved

Y = [0, 0, 1, 1, 1]

# Create Model
model = GaussianNB()

# Train Model
model.fit(X, Y)

# Test Data
test = [[45000, 720]]

# Prediction
result = model.predict(test)

# Output
print("Customer Details:", test)

print("\nLoan Prediction:")

if result[0] == 1:
    print("Loan Approved")

else:
    print("Loan Rejected")
