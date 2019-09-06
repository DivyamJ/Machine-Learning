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

words = pd.read_csv('words.txt', header=None, names=['word'])
words.index +=1

traindata = pd.read_csv('traindata.txt', header=None, sep = '\t', names=['docld','wordld'])
traindata.index +=1
trainlabel = pd.read_csv('trainlabel.txt', header=None, sep = '\t', names=['category'])
trainlabel.index +=1
testdata = pd.read_csv('testdata.txt', header=None, sep = '\t', names=['docld','wordld'])
testdata.index +=1
testlabel = pd.read_csv('testlabel.txt', header=None, sep = '\t', names=['category'])
testlabel.index +=1

dataset_train = pd.DataFrame(index=trainlabel.index, columns = words.index)
dataset_train= dataset_train.fillna(0)
dataset_train['label'] = trainlabel['category']

for i in traindata.index:
    dataset_train[traindata['wordld'][i]][traindata['docld'][i]]=1 #Training Dataset ready
    
dataset_test = pd.DataFrame(index=testlabel.index, columns = words.index)
dataset_test= dataset_test.fillna(0)
dataset_test['label'] = testlabel['category']

for i in testdata.index:
    dataset_test[testdata['wordld'][i]][testdata['docld'][i]]=1  #Test Dataset ready

#Calculating Information Gain

def info_gain (df):
    
    #df=dataset_train.loc[:,[1,'label']]
    mat = pd.DataFrame(index=[0,1],columns=[1,2])
    mat = mat.fillna(0)
     
    for i in range(len(df)):
        mat[df.iloc[i,1]][df.iloc[i,0]]+=1   
    
    x = mat.sum(axis=0) 
    entropy_before =  -x[1]/(x[1]+x[2])*math.log2(x[1]/(x[1]+x[2]))-x[2]/(x[1]+x[2])*math.log2(x[2]/(x[1]+x[2]))

        
    val=0
    for i in range(len(mat)):
        a=mat.iloc[i,0]
        b=mat.iloc[i,1]
        
        if a!=0 and b!=0:
            val = val-a*(math.log2(a/(a+b)))-b*(math.log2(b/(a+b)))
        
    entropy_after = val/len(df)
    
    return entropy_before-entropy_after
    
#Spliting Dataset 
    
def split (feature, dataset):

    B = []
    for j in [0,1]:
        B.append(dataset[dataset[feature]==j])
    
    return B

#Making Decison Tree using Information Gain

def best_split_info(feature_set, data, level, temp, depth):
    
    if(len(data[data.iloc[:,-1]==1])==len(data)):
        print(':alt.atheism', end='')
        temp['label'] = 1
    elif(len(data[data.iloc[:,-1]==2])==len(data)):
        print(':comp.graphics', end='')
        temp['label'] = 2

    elif len(feature_set) == 0 or level==depth:
        if(len(data)>0):
            count_1 = len(data[data.iloc[:,-1]==1])
            if count_1 >= len(data)-count_1:
                print(':alt.atheism', end='')
                temp['label'] = 1
            else:
                print(':comp.graphics', end='')
                temp['label'] = 2
    elif len(data)==0:
        pass
    
    else:
    
        A =[]
        for i in feature_set:
            A.append(info_gain(data.loc[:,[i,'label']]))
            
        best_feature = feature_set[A.index(max(A))]
        B = split(feature_set[A.index(max(A))], data)
        
        temp[best_feature] = {}
        temp = temp[best_feature]
        
        feature_set.remove(best_feature)
        
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
            print(str(level)+'Hi its Divyam') 
            best_split_info(feature_set, B[i], level+1, temp[i], depth)
            
#Predicting label of Test Data        
        
def predict (row, tree):
    while(list(tree.keys())[0]!='label'):
        x=row.loc[:,list(tree.keys())[0]]
        x=x.iloc[0]
        tree = tree[list(tree.keys())[0]]
        tree = tree[x]
    return tree['label']
            
#Using Information Gain

dep = int(input('What is the maximum depth? '))

print('\n\n---Decision Tree using Information Gain---', end='')       
model = {}
temp=model  
        
best_split_info(list(dataset_train.columns[:-1]), dataset_train, 0, temp, dep)

    




    
    


