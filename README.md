# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R.Avanesh
RegisterNumber:25018356 

```
```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Read CSV file
# -------------------------------
df = pd.read_csv("marks.csv")

# Display first few rows
print(df.head())

# -------------------------------
# Separate variables
# -------------------------------
X = df[['Hours']]   # Independent variable
y = df['Scores']     # Dependent variable

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation Metrics
# -------------------------------
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²  :", r2_score(y_test, y_pred))

# -------------------------------
# Visualization
# -------------------------------
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression using CSV File")
plt.legend()
plt.show()

```

## Output:
![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/e74202dc-3e2f-48db-8d2f-84dc9aa4cdeb)

![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/a9682a0c-6b81-46e3-b071-2e4a3aa50a1d)

![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/7af1094c-e890-4a53-a684-00afb1f9b0f0)

![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/1c580e33-464f-47ad-9eff-4851bf17a91a)

![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/a9411b7e-a555-48a3-952a-10b1e61726ae)

![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/91beb994-6872-4bf0-8adb-0d31dcee50e1)

![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/9d5fbbb4-70e3-4a8a-bffb-6b37bff77cbc)

![image](https://github.com/harini1006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497405/a21e6f27-be9e-4879-9afa-b52b0290271c)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
