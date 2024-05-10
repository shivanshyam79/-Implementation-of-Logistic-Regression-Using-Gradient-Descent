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
## Array value of X
![270399211-e0d2116b-fafc-4458-a5ba-edabd4cebf54](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/a98dbc59-341d-4d16-8773-eff60cad6069)
## Array value of Y
![270399296-2574b577-9872-439c-9375-eb80aaa412b6](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/7e966e30-9e1a-4400-8872-7260ccdd3899)
## Score graph
![270399357-4235eee2-5337-4ce8-baec-a66aadf16f0a](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/5e0cd063-e576-4759-996a-f72f1cfeadbc)
## Sigmoid function graph
![270399416-a20ecae5-99f0-43ad-8a57-5ca4b5320569](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/c6b09bc0-c759-483c-8e02-8f4ebd434cd1)
## X train grad value
![270399513-0a671b8d-5898-4e69-9a42-6806eb05cd99](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/22cd9d4c-bede-4ef5-8c7e-6f3258fbb125)
## Y train grad value
![270399594-c5bad725-4da0-4c89-aaf3-a653995f9ad9](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/059d417a-c5f8-457c-9060-979b9111ad81)
Regression value
![270399669-b69d76a9-4fec-4f65-9a56-7ea9b4e47184](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/43961d8a-094f-44ee-ab5b-361aa32a4660)
Decision boundary graph
![270399736-ca4cf70b-3faf-4f79-b267-ac6ea9b85470](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/4ff0b414-30b3-436d-9cb6-e3be682e914d)
probablity value
![270399802-e2c1c8b0-3689-4644-bee1-44f77f212884](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/1ca2aad8-5c1f-44b8-a433-74ede6d22d06)
Prediction value of graph
![270399849-06245660-99ae-4c65-8d3b-ce804e3afc96](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/84cfb458-fd13-457f-9581-0b2f1e3b384d)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
