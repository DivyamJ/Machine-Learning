"""
Assignment 1: Linear Regression
Part 2

@author: Divyam Jain
         17HS20047
"""
#Importing some essential libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Generating a synthetic data set

n = 10  #n is number of instances
X = []  #X is the matrix of independent variable

for i in range(0,n):     #generates independent variable matrix
    X.append(i/(n-1))    #uniformly spread over [0,1]
    
Y_raw = [ np.sin(2*np.pi*elem) for elem in X]  #computes sin[2*pi*x]

noisedata = np.loadtxt("noisedata.csv") #for the same noise generated in Part 1 of this question

Y = [] #Y is the dependent variable matrix
for i in range(0,len(X)):
    Y.append(Y_raw[i]+noisedata[i])   #adds noise to sin(2*pi*x) to get target Y

#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 36) 

#Applying Gradient Descent

def getparameters(degree):
    alpha = 0.05    #learning rate = 0.05
    phi_x = np.array([[elem**i for i in range(0,degree+1)] for elem in X_train])   #constructing a 2-D array 
    phi_x = phi_x.T
    W= np.zeros((1,degree+1))
    
    for j in range(0,1000):
        arr = np.subtract(np.matmul(W, phi_x), Y_train)
        
        A = [ (np.matmul(arr, phi_x[i,:].T))[0] for i in range(0,degree+1) ]
        A = np.reshape(A,(1,len(A)))
        
        W = np.subtract(W, (alpha/len(X_train))*A)
        
    return W

W_final = []
Test_error = []
Train_error = []
for k in range(1,10):
    W = getparameters(k)
    W_final.append(W)
    
    phi_y = np.array([[elem**i for i in range(0,k+1)] for elem in X_test])    
    phi_y = phi_y.T
    
    ind_error = np.subtract(np.matmul(W,phi_y), Y_test) 
    total_errorsq = float((np.matmul(ind_error,ind_error.T))[0])
    Test_error.append((1/(2*len(X_test)))*(total_errorsq))
    
    phi_x = np.array([[elem**i for i in range(0,k+1)] for elem in X])
    phi_x = phi_x.T
    y_pred = np.matmul(W,phi_x)
    
    phi_xtrain = np.array([[elem**i for i in range(0,k+1)] for elem in X_train])
    phi_xtrain = phi_xtrain.T
    y_predtrain = np.matmul(W,phi_xtrain)
    
    ind_error_train = np.subtract(y_predtrain, Y_train)
    total_errorsq = float((np.matmul(ind_error_train,ind_error_train.T))[0])
    Train_error.append((1/(2*len(X_train)))*(total_errorsq))
    
    plt.scatter(X_train,Y_train,color='red')
    
    a1=[]
    for l in range(0,len(X)):
        a1.append(y_pred[0,l])
        
    plt.plot(X, a1, color = 'blue')
    plt.title('Visualising the curve degree = %i' %k)
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.show()
   
plt.plot([1,2,3,4,5,6,7,8,9], Train_error, color = 'red')
plt.plot([1,2,3,4,5,6,7,8,9], Test_error, color = 'blue')
plt.title('Training error(red) and Test Error(blue) for degrees 1 to 9')
plt.xlabel('Degree of polynomial')
plt.ylabel('Cost')
plt.show()




    
    
    


    










  