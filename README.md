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
### Data Set:
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/65b41726-0120-4a4b-b40a-d43e248823d9)

### Head Value
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/240b2352-15b4-43d4-a1e9-5bf7cff9c73c)

### Tail Value
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/74f38c0c-076a-4468-b7ec-beda04215aa7)

### X and Y Values
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/93dc266d-8193-46b9-aacb-f235729cd7eb)

### Predication values of X and Y
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/1c6dbe39-5ea4-4fad-9f86-4115bcb5f5b6)

### MSE,MAE and RMSE
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/7bbe9578-3bc9-4420-8f78-1c84e39fe21f)

### Training Set
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/8719ffc5-5bc7-4a8e-9002-42f8be22da89)


### Testing Set:
![image](https://github.com/Safeeq-Fazil/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680361/2afee5df-91f6-4421-b922-c7fa554885fd)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
