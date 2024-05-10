# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## step 1. Initialize Parameters: Set initial values for the weights (w) and bias (b).
## step 2. Compute Predictions: Calculate the predicted probabilities using the logistic function.
## step 3. Compute Gradient: Compute the gradient of the loss function with respect to w and b.
## step 4. Update Parameters: Update the weights and bias using the gradient descent update rule. Repeat 
## step 5 stop
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
![199067370-21f6e068-3851-4596-bad3-35dc02d079a6](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/5ddf7e50-1ff5-4278-b756-6cc4552e9c7d)
![199067368-23904c41-d2d1-4e62-83d0-29a65b810abe](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/2cfd43b9-0e8e-4d85-a246-6e8936858ab3)
![199067364-67b76106-9b8d-4758-a093-ec7e7f4b2d32](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/f139e202-51ff-4711-84cc-0670957816c2)
![199067359-63750fd2-98e8-438d-a32c-ae84cb1d27e4](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/6d74e4db-36bb-4f89-8b2d-4169e4996cb7)
![199067356-69f818c1-d425-48e8-beb3-281e00b6ecba](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/100935c3-8826-4b24-acc6-47e8035e8246)
![199067352-588a14f1-b111-4fc6-801d-acb4fd847520](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/6f4af397-4804-4244-b97f-7b5083bfebd6)
![199067351-3e334116-ed7b-441b-93e6-20737be81d24](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/8e222498-9f53-48b8-9e56-b37982664d7e)
![199067346-56d58684-54aa-478a-98ac-f9841f1b846e](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/9b68334c-ac06-49c4-ac24-c1ef89d7c625)
![199067342-fbdbcd76-c1d0-4fb3-95cb-d847e85e0d51](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/d817b69a-066b-4899-9dfa-3336700438d9)

![199067377-9f1bdbbb-7868-4b11-8bed-f98680735040](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/ccb20051-cd4b-4a10-a58d-e336443e4eb7)


### Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

