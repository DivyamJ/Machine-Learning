"""
Assignment 1: Linear Regression
Part 4

@author: Divyam Jain
         17HS20047
"""
#Importing some essential libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Generating a synthetic data set

n = 100  #n is number of instances
X = []  #X is the matrix of independent variable

for i in range(0,n):     #generates independent variable matrix
    X.append(i/(n-1))    #uniformly spread over [0,1]
    
Y_raw = [ np.sin(2*np.pi*elem) for elem in X]  #computes sin[2*pi*x]

noise = np.random.normal(loc=0,scale=0.3,size=n) #random noise generator

Y = [] #Y is the dependent variable matrix
for i in range(0,len(X)):
    Y.append(Y_raw[i]+noise[i])   #adds noise to sin(2*pi*x) to get target Y

#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 36) 

#Applying Gradient Descent
'''number of instances chosen in dataset = 100 and degree of polyomial as 9'''


phi_x = np.array([[elem**i for i in range(0,10)] for elem in X_train])   #constructing a 2-D array 
phi_x = phi_x.T
W = np.zeros((1,10))

W_finalcost1=[]
Test_errorcost1=[]
#Cost Function 1

for alpha in [0.025,0.05,0.1,0.2,0.5]:                 #alpha is learning rate

    for j in range(0,1000):
        arr = np.subtract(np.matmul(W, phi_x), Y_train)
        arr = np.sign(arr)                              #signum function as derivative of mod function   
        A = [ (np.matmul(arr, phi_x[i,:].T))[0] for i in range(0,10) ]
        A = np.reshape(A,(1,len(A)))
                
        W = np.subtract(W, (alpha/(2*len(X_train)))*A)
        
    W_finalcost1.append(W)
    
    phi_y = np.array([[elem**i for i in range(0,10)] for elem in X_test])    
    phi_y = phi_y.T
        
    ind_error = np.subtract(np.matmul(W,phi_y), Y_test) 
    total_errormod = np.sum(np.absolute(ind_error))
    Test_errorcost1.append((1/(2*len(X_test)))*(total_errormod))


W = np.zeros((1,10))
W_finalcost2=[]
Test_errorcost2=[]
#Cost Function 2

for alpha in [0.025,0.05,0.1,0.2,0.5]:                 #alpha is learning rate

    for j in range(0,1000):
        arr = np.subtract(np.matmul(W, phi_x), Y_train)
        arr = [ l**3 for l in arr]  
        arr = np.reshape(np.array(arr),(1,len(X_train)))
        A = [ (np.matmul(arr, phi_x[i,:].T))[0] for i in range(0,10) ]
        A = np.reshape(A,(1,len(A)))
                
        W = np.subtract(W, ((2*alpha)/(len(X_train)))*A)
        
    W_finalcost2.append(W)
    
    phi_y = np.array([[elem**i for i in range(0,10)] for elem in X_test])    
    phi_y = phi_y.T
        
    ind_error = np.subtract(np.matmul(W,phi_y), Y_test)
    err = [l**4 for l in ind_error]
    err = np.reshape(err,(1,len(X_test)))
    
    total_error4 = np.sum(err)
    Test_errorcost2.append((1/(2*len(X_test)))*(total_error4))


plt.plot([0.025,0.05,0.1,0.2,0.5], Test_errorcost1, color = 'red' )
plt.title('Test error vs alpha(learning rate) for cost function 1(mod)')
plt.xlabel('Alpha(learning rate)')
plt.ylabel('Cost')
plt.show()

plt.plot([0.025,0.05,0.1,0.2,0.5], Test_errorcost2, color = 'red' )
plt.title('Test error vs alpha(learning rate) for cost function 2(fourth power)')
plt.xlabel('Alpha(learning rate)')
plt.ylabel('Cost')
plt.show()










  