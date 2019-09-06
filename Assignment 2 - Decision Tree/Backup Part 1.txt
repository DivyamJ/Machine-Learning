"""
Assignment 2: Decision Trees
Part 1

@author: Divyam Jain
         17HS20047
"""
#Importing some essential libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Importing the dataset

dataset_train = pd.read_excel('dataset for part 1.xlsx', sheetname = 'Training Data')
dataset_test = pd.read_excel('dataset for part 1.xlsx', sheetname = 'Test Data')

#Calculating Information Gain

def info_gain (df):
      
    unique = pd.unique(df.iloc[:,0])
    mat = pd.DataFrame(index=unique,columns=['no','yes'])
    mat = mat.fillna(0)
     
    for i in range(len(df)):
        mat[df.iloc[i,1]][df.iloc[i,0]]+=1   
    
    x = mat.sum(axis=0)    
    entropy_before = -x[0]/(x[0]+x[1])*math.log2(x[0]/(x[0]+x[1]))-x[1]/(x[0]+x[1])*math.log2(x[1]/(x[0]+x[1]))
    
    val=0
    for i in range(len(mat)):
        a=mat.iloc[i,0]
        b=mat.iloc[i,1]
        
        if a!=0 and b!=0:
            val = val-a*(math.log2(a/(a+b)))-b*(math.log2(b/(a+b)))
        
    entropy_after = val/len(df)
    
    return entropy_before-entropy_after

#Calculating Gini Index

def gini (df):
    
    unique = pd.unique(df.iloc[:,0])
    mat = pd.DataFrame(index=unique,columns=['no','yes'])
    mat = mat.fillna(0)
    
    for i in range(len(df)):
        mat[df.iloc[i,1]][df.iloc[i,0]]+=1
        
    val=0
    for i in range(len(mat)):
        a=mat.iloc[i,0]
        b=mat.iloc[i,1]
        if (a+b)!=0:
            val = val + (a+b)*(1-(a/(a+b))**2-(b/(a+b))**2)
        
    return val/len(df)

#Spliting Dataset 
    
def split (feature, dataset):

    unique = pd.unique(dataset.loc[:,feature])
    
    B = []
    for j in unique:
        B.append(dataset[dataset[feature]==j])
    
    return B
        
#Making Decison Tree using Information Gain

def best_split_info(feature_set, data, level, temp):
    
    if(len(data[data.iloc[:,-1]=='no'])==len(data)):
        print(':no', end='')
        temp['profitable'] = 'no'
    elif(len(data[data.iloc[:,-1]=='yes'])==len(data)):
        print(':yes', end='')
        temp['profitable'] = 'yes'

    elif len(feature_set) == 0:
        if(len(data)>0):
            count_no = len(data[data.iloc[:,-1]=='no'])
            if count_no >= len(data)-count_no:
                print(':no', end='')
                temp['profitable'] = 'no'
            else:
                print(':yes', end='')
                temp['profitable'] = 'yes'
    elif len(data)==0:
        pass
    
    else:
    
        A =[]
        for i in feature_set:
            A.append(info_gain(data.loc[:,[i,'profitable']]))
            
        best_feature = feature_set[A.index(max(A))]
        B = split(feature_set[A.index(max(A))], data)
        unique = list(pd.unique(data.loc[:,best_feature]))
        
        temp[best_feature] = {}
        temp = temp[best_feature]
        
        feature_set.remove(best_feature)
        
        for i in unique:
                
            if level>=1:
                if len(B[unique.index(i)])!=0:
                    print('\n')
                    for j in range(1,level):
                        print('\t', end ='')
                    print('|', end='')
                    print(best_feature + '=', end='')
                    print(i, end='')
                    temp[i]={}
                    
            else:
                if len(B[unique.index(i)])!=0:
                    print('\n')
                    print(best_feature + '=', end='')
                    print(i, end='')
                    temp[i]={}
                
            best_split_info(feature_set, B[unique.index(i)], level+1, temp[i])


#Making Decison Tree using Gini Index

def best_split_gini(feature_set, data, level, temp):
    
    if(len(data[data.iloc[:,-1]=='no'])==len(data)):
        print(':no', end='')
        temp['profitable'] = 'no'
    elif(len(data[data.iloc[:,-1]=='yes'])==len(data)):
        print(':yes', end='')
        temp['profitable'] = 'yes'

    elif len(feature_set) == 0:
        if(len(data)>0):
            count_no = len(data[data.iloc[:,-1]=='no'])
            if count_no >= len(data)-count_no:
                print(':no', end='')
                temp['profitable'] = 'no'
            else:
                print(':yes', end='')
                temp['profitable'] = 'yes'
    elif len(data)==0:
        pass
    
    else:
    
        A =[]
        for i in feature_set:
            A.append(gini(data.loc[:,[i,'profitable']]))
            
        best_feature = feature_set[A.index(min(A))]
        B = split(feature_set[A.index(min(A))], data)
        unique = list(pd.unique(data.loc[:,best_feature]))
        
        temp[best_feature] = {}
        temp = temp[best_feature]
        
        feature_set.remove(best_feature)
        
        for i in unique:
                
            if level>=1:
                if len(B[unique.index(i)])!=0:
                    print('\n')
                    for j in range(1,level):
                        print('\t', end ='')
                    print('|', end='')
                    print(best_feature + '=', end='')
                    print(i, end='')
                    temp[i]={}
                    
            else:
                if len(B[unique.index(i)])!=0:
                    print('\n')
                    print(best_feature + '=', end='')
                    print(i, end='')
                    temp[i]={}
                
            best_split_gini(feature_set, B[unique.index(i)], level+1, temp[i])
            
#Predicting label of Test Data        
        
def predict (row, tree):
    while(list(tree.keys())[0]!='profitable'):
        x=row.loc[:,list(tree.keys())[0]]
        x=x.iloc[0]
        tree = tree[list(tree.keys())[0]]
        tree = tree[x]
    return tree['profitable']

#Using Information Gain

print('\n\n---Decision Tree using Information Gain---', end='')       
model_info = {}
temp=model_info   
       
best_split_info(list(dataset_train.columns[:-1]), dataset_train, 0, temp)

#Using Gini Index

print('\n\n\n---Decision Tree using Gini Index---', end='')  
model_gini = {}
temp=model_gini   
       
best_split_gini(list(dataset_train.columns[:-1]), dataset_train, 0, temp)


    
#Finding the output of Test Data using Information Gain Trained Model  

pred_info = []

for i in dataset_test.index:
    b=dataset_test.loc[i,dataset_test.columns!='profitable']
    b=pd.Series.to_frame(b)
    b=b.transpose()
    pred_info.append(predict(b,model_info))
    
#Finding accuracy of output using Information Gain Trained Model
    
count = 0
for i in range(len(pred_info)):
    if dataset_test.iloc[i,-1]==pred_info[i]:
        count+=1
acc_info = count/len(pred_info)*100

#Finding Root Node Information Gain

A =[]
for i in list(dataset_train.columns[:-1]):
    A.append(info_gain(dataset_train.loc[:,[i,'profitable']]))
root_info = max(A) 
    
#Finding the output of Test Data using Gini Index Trained Model  
    
pred_gini = []

for i in dataset_test.index:
    b=dataset_test.loc[i,dataset_test.columns!='profitable']
    b=pd.Series.to_frame(b)
    b=b.transpose()
    pred_gini.append(predict(b,model_gini))
    
#Finding accuracy of output using Gini Index Trained Model
    
count = 0
for i in range(len(pred_gini)):
    if dataset_test.iloc[i,-1]==pred_gini[i]:
        count+=1
acc_gini = count/len(pred_gini)*100

#Finding Root Node Gini Index

a=len(dataset_train[dataset_train['profitable']=='yes'])
b=len(dataset_train)-a
root_gini = (1-(a/(a+b))**2-(b/(a+b))**2)


#Implementation using SciKit Learn

from sklearn.tree import DecisionTreeClassifier

#Converting into categorical variables
 
labels=dataset_train.iloc[:,-1].values
for j in labels:
    if j=='yes':
        j=1
    else:
        j=0
features=dataset_train.iloc[:,:-1].values
for j in range(len(features)):
    for k in [0,1,3]:
        if features[j][k]=='low':
            features[j][k]=0
        elif features[j][k]=='med':
            features[j][k]=1
        elif features[j][k]=='high':
            features[j][k]=2
        elif features[j][k]=='no':
            features[j][k]=0
        else:
            features[j][k]=1
            
testing = dataset_test.iloc[:,:-1].values
for j in range(len(testing)):
    for k in [0,1,3]:
        if testing[j][k]=='low':
            testing[j][k]=0
        elif testing[j][k]=='med':
            testing[j][k]=1
        elif testing[j][k]=='high':
            testing[j][k]=2
        elif testing[j][k]=='no':
            testing[j][k]=0
        else:
            testing[j][k]=1

#Using Information Gain

classifier_info = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_info.fit(features,labels)

pred_infosci = classifier_info.predict(testing)

count = 0
for i in range(len(pred_infosci)):
    if dataset_test.iloc[i,-1]==pred_infosci[i]:
        count+=1
acc_infosci = count/len(pred_infosci)*100


#Using Gini Index

classifier_gini = DecisionTreeClassifier(criterion='gini', random_state=0)
classifier_gini.fit(features,labels)

pred_ginisci = classifier_gini.predict(testing)

count = 0
for i in range(len(pred_ginisci)):
    if dataset_test.iloc[i,-1]==pred_ginisci[i]:
        count+=1
acc_ginisci = count/len(pred_ginisci)*100 

    
    





    
    
        





