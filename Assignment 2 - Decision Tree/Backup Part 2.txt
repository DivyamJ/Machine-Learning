"""
Assignment 2: Decision Trees
Part 2

@author: Divyam Jain
         17HS20047
"""

#Importing some essential libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Importing the dataset

words = np.loadtxt('words.txt', dtype='str')

traindata = np.loadtxt('traindata.txt')
trainlabel = np.loadtxt('trainlabel.txt')
testdata = np.loadtxt('testdata.txt')
testlabel = np.loadtxt('testlabel.txt')


dataset_train = np.zeros((len(trainlabel),len(words)+1),dtype=int)
dataset_train[:,-1] = trainlabel[:]-1
for i in range(len(traindata)):
    dataset_train[int(traindata[i,0])-1,int(traindata[i,1])-1]+=1
    
dataset_test = np.zeros((len(testlabel),len(words)+1),dtype=int)
dataset_test[:,-1] = testlabel[:]-1
for i in range(len(testdata)):
    dataset_test[int(testdata[i,0])-1,int(testdata[i,1])-1]+=1

#Calculating Information Gain

def info_gain (df):

    mat=np.zeros((2,2))
    for i in range(len(df)):
        mat[df[i,0],df[i,1]]+=1   
    
    x = np.sum(mat,axis=0) 
    entropy_before =  -x[0]/(x[0]+x[1])*math.log2(x[0]/(x[0]+x[1]))-x[1]/(x[0]+x[1])*math.log2(x[1]/(x[0]+x[1]))

        
    val=0
    for i in range(len(mat)):
        a=mat[i,0]
        b=mat[i,1]
        
        if a!=0 and b!=0:
            val = val-a*(math.log2(a/(a+b)))-b*(math.log2(b/(a+b)))
        
    entropy_after = val/len(df)
    
    return entropy_before-entropy_after
    
#Spliting Dataset 
    
def split (feature, dataset):

    B = []
    for j in [0,1]:
        B.append(dataset[dataset[:,feature]==j])
    
    return B

#Making Decison Tree using Information Gain

def best_split_info(feature_set, data, level, temp, depth):
    
    if(len(data[data[:,-1]==0])==len(data)):
        print(':alt.atheism', end='')
        temp['label'] = 0
    elif(len(data[data[:,-1]==1])==len(data)):
        print(':comp.graphics', end='')
        temp['label'] = 1

    elif level==depth:
        if(len(data)>0):
            count_0 = len(data[data[:,-1]==0])
            if count_0 >= len(data)-count_0:
                print(':alt.atheism', end='')
                temp['label'] = 0
            else:
                print(':comp.graphics', end='')
                temp['label'] = 1
    elif len(data)==0:
        pass
    
    else:
    
        A =[]
        for i in feature_set:
            A.append(info_gain(data[:,[i,-1]]))
            
        best_feature = A.index(max(A))
        B = split(best_feature, data)
        
        temp[best_feature] = {}
        temp = temp[best_feature]
        
        #feature_set.remove(best_feature)
        
        for i in [0,1]:
                
            if level>=1:
                if len(B[i])!=0:
                    print('\n')
                    for j in range(1,level):
                        print('\t', end ='')
                    print('|', end='')
                    print(best_feature,'=',sep='', end='')
                    print(i, end='')
                    temp[i]={}
                    
            else:
                if len(B[i])!=0:
                    print('\n')
                    print(best_feature,'=',sep='', end='')
                    print(i, end='')
                    temp[i]={}
                    
            best_split_info(feature_set, B[i], level+1, temp[i], depth)
            
#Predicting labels of Data        
        
def predict (row, tree):
    while(list(tree.keys())[0]!='label'):
        x=row[list(tree.keys())[0]]
        tree = tree[list(tree.keys())[0]]
        tree = tree[x]
    return tree['label']
            
#Using Information Gain

dep = int(input('What is the required depth? '))

print('\n\n---Decision Tree using Information Gain---', end='')       
model = {}
temp=model  
feature_set=np.arange(0,len(words))  
best_split_info(feature_set, dataset_train, 0, temp, dep)

#Getting Accuracies for varied depths

max_depth = 25
acc_train=[]
acc_test=[]

for j in range(1,max_depth):
    
    print('\n\n---Decision Tree using Information Gain---  max_depth =',j, end='')
    best_split_info(feature_set, dataset_train, 0, temp, j)
    
    #Finding Training Data accuracy 

    pred = []

    for k in range(len(dataset_train)):
        pred.append(predict(dataset_train[k,:-1],model))
    
    count = 0
    for i in range(len(pred)):
        if dataset_train[i,-1]==pred[i]:
            count+=1
    accuracy = count/len(pred)*100
    
    acc_train.append(accuracy)
    
    #Finding Test Data accuracy 

    pred = []

    for k in range(len(dataset_test)):
        pred.append(predict(dataset_test[k,:-1],model))
    
    count = 0
    for i in range(len(pred)):
        if dataset_test[i,-1]==pred[i]:
            count+=1
    accuracy = count/len(pred)*100
    
    acc_test.append(accuracy)
    
#Plotting graphs
    
plt.plot(np.arange(1,max_depth), acc_train, color = 'blue')
plt.plot(np.arange(1,max_depth), acc_test, color = 'red')
plt.title('Training(Blue) and Test(Red) Accuracy for various maximum depths')
plt.xlabel('Maximium Depth')
plt.ylabel('Accuracies')
plt.show()

#Using Scikit Learn

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(dataset_train[:,:-1],dataset_train[:,-1])

scipred = classifier.predict(dataset_test[:,:-1])

count = 0
for i in range(len(scipred)):
    if dataset_test[i,-1]==scipred[i]:
        count+=1
sciaccuracy = count/len(scipred)*100
    

    
    
    

    




    
    


