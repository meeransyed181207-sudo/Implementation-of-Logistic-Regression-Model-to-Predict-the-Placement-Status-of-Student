# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the student placement dataset and preprocess the data (clean, encode, and scale features).
2. Split the dataset into training and testing sets.
3. Train a Logistic Regression model using the training data.
4. Predict the placement status and evaluate the model using accuracy metrics.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SYED MEERAN A
RegisterNumber:  25008812
*/
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:/Users/acer/Downloads/Placement_Data.csv")
data.drop("sl_no", axis=1, inplace=True)

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
data = pd.get_dummies(data, drop_first=True)
X = data.drop('status', axis=1).values
y = data['status'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test  = (X_test  - X_test.mean(axis=0))  / X_test.std(axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.1      # increased
epochs = 3000            # increased

for _ in range(epochs):
    linear = np.dot(X_train, weights) + bias
    y_pred = sigmoid(linear)

    dw = (1 / len(y_train)) * np.dot(X_train.T, (y_pred - y_train))
    db = (1 / len(y_train)) * np.sum(y_pred - y_train)

    weights -= learning_rate * dw
    bias -= learning_rate * db

def predict(X):
    linear = np.dot(X, weights) + bias
    return np.where(sigmoid(linear) >= 0.5, 1, 0)

y_predicted = predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predicted) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predicted))

print("\nClassification Report:")
print(classification_report(y_test, y_predicted))
```

## Output:
<img width="773" height="355" alt="image" src="https://github.com/user-attachments/assets/5b6e424e-768d-4628-846c-7b818e71f7d4" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
