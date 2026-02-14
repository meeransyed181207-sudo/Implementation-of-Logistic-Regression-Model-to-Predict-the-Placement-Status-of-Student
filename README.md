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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data.csv")

print(data.head())

data.drop('sl_no', axis=1, inplace=True)

data['salary'].fillna(0, inplace=True)
le = LabelEncoder()
categorical_columns = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


```

## Output:
<img width="755" height="295" alt="image" src="https://github.com/user-attachments/assets/682b5dee-0f8a-4a55-8ab3-cae4e2b3155a" />
<img width="597" height="364" alt="image" src="https://github.com/user-attachments/assets/33292411-99f7-47ca-9018-90b5b93de518" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
