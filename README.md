# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Set initial values for the weights (w) and bias (b).
2. Compute Predictions: Calculate the predicted probabilities using the logistic function.
3. Compute Gradient: Compute the gradient of the loss function with respect to w and b.
4. Update Parameters: Update the weights and bias using the gradient descent update rule. Repeat steps 2-4 until convergence or a maximum number of iterations is reached.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:shyam R
RegisterNumber: 212223040200
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/SEC/Downloads/Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/9a6375f1-3276-47e1-87dc-efa2c319789b)

![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/c0a5d9ab-ad43-491b-bc8d-ae1b84e7bd2a)

![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/7a689516-72cc-45ff-aaad-1588e8744fb0)

![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/88e2a527-7fff-4f78-834e-2cd0a8c480f9)

![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/fd8f803e-49f6-4f59-9ced-800c5fb9fd2a)

![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/0215ca6b-21d1-425d-a65b-2e2d48242391)

![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/ea873136-4387-4c41-96ae-dca9d1a6daa9)

![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/f620fb51-026f-4602-8930-68e747a1be2c)
![image](https://github.com/Praveenanagaraji22/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393514/46e1a8c1-c2d3-4a5d-9b23-086973db5e23)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
