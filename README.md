# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient. 


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: shyam R
RegisterNumber:  212223040200
Program to implement the the Logistic Regression Using Gradient Descent.
# Developed by: Sri Varshan P
# RegisterNumber:  212222240104
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]
print("Array of X") 
X[:5]
print("Array of y") 
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()
def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)
def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)
prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)
def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
print("Prediction value of mean")
np.mean(predict(res.x,X)==y)

*/
```

## Output:
### ![270415074-4903db55-4842-42e4-987e-cec0ace878e4](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/6cc89937-043a-4a11-b26e-7d05b530ce38)
### ![270415116-8b54bce9-0d54-4560-9768-0e2c1c682851](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/ce2f0edb-f9f4-477b-bda0-d9133d3da622)
### ![270415165-1bc0fe62-269a-4f78-b6c8-969293a6b29c](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/05a3ca7b-c79a-417e-a9cf-8fa4baab0140)
### ![270415220-426970ab-a3eb-4584-a5de-4d05c41ad825](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/7111dbf3-17df-45fa-89ac-7eade3eaf720)
### ![270415252-d8a88daf-6410-482b-9aaa-8cc02e674fcc](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/24072eb0-fb5e-4f47-af01-6d75e8c95d24)
### ![270415281-89e30d97-fc7b-4370-9df7-1757b3a22d07](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/58a5fc82-8683-441e-af6f-e126d3e8f47b)
### ![270415307-69e18e12-9fd3-4394-b94d-6559206ba9b7](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/607caef2-dbc5-4c52-9cee-a24faeef73f1)
### ![270415362-d9b28c22-f3be-4176-b115-7c003f831ed2](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/ec409b30-2c5b-4f24-8898-e5d9f95792d2)
### ![270415422-d4fdaf1f-0242-49ef-92bb-38899754cd96](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/6e6943a2-4046-462f-aff8-88eb6e7efdf5)
### ![270415463-eb7aacb6-9f8c-4795-ab98-b015c60456c6](https://github.com/shivanshyam79/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151513860/42b19953-b2b1-4a56-ada3-1fa6dd714eda)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

