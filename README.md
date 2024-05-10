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
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MAHESH RAJ PUROHIT J
RegisterNumber:212222240058

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data

plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
```

## Output:
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/12de31bc-efe5-468e-b300-92381a2d6c71)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/a2d17d87-ab45-4b9d-a999-cbb87a9f6d90)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/6cd09eda-6833-41eb-bf9f-8f25d5580894)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/19d37f6d-c10b-43b2-8f08-57e1aa41ff3e)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/f07ee377-a128-45e5-8a02-870e72216dd2)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/6c95449f-3509-4afd-a3ab-3da0005063c7)
### values of MSE, MAE, RMSE:
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/42a30f4a-4bfa-45b7-a7d9-7d262ee43dc0)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/5f1d8d44-f671-4ecf-93fe-ad5fe620045f)
![image](https://github.com/Kousalya22008930/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389108/4511bbb1-d051-49ef-97f3-434130b651ca)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
