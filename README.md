# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner



## Algorithm
~~~
1. Import the required Libraries.
2.Import the csv file.
3.Declare X and Y values with respect to the dataset.
4.Plot the graph using the matplotlib library.
5.Print the plot.
6.End the program
~~~



## Program:
```
~~~
#developed by : S Adithya Chowdary.
#Reference number: 212221230100.
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
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
y_pred
y_test
plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()
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
```
## Output:
![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/f14e4cb3-c4f3-49f5-abf1-a5f9ad544f10)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/ae92fdf7-63db-43a2-bb16-56609e6f86ee)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/0a1510e7-ea6f-40c8-ae8e-2fafcc41eb06)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/6791134e-c3e6-4dbc-a964-d7addfd29ff7)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/47613689-a1a6-4a0f-bbef-572211ce556d)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/0850b263-1004-44d9-bcb4-683b3d1d8967)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/14de8910-7ad5-4ab1-95cd-c68a79850f17)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/830fc6ca-8802-4dc6-bf92-bb2da4b9a478)

![image](https://github.com/Adithya-Siddam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/93427248/8e2826e2-988a-44bc-8b21-d1e81f564120)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
