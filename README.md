# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```'''
Program to implement the simple linear regression model for predicting the marks scored.
Developed by : R.Avanesh
RegisterNumber : 25018356


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
print("HEAD:")
print(df.head())
print("TAIL:")
print(df.tail())
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
import matplotlib.pyplot as plt
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
MSE = mean_squared_error(Y_test,Y_pred)
print('MSE = ',MSE)
MAE = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',MAE)
RMSE=np.sqrt(MSE)
print("RMSE = ",RMSE)
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```

## Output:
'''image
image
image
image
image
image
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
