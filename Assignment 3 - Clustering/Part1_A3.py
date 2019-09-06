"""
Assignment 3: Clustering
Part 1

@author: Divyam Jain
         17HS20047
"""
#Importing some essential libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset

dataset = pd.read_csv("AAAI.csv")

#Making the topics array preserving the index

n=len(dataset)
A=[]
for i in range(n):
    A.append(dataset['Topics'][i].split('\n'))
    
#Computing the jaccard coefficients in a matrix

jaccard = np.zeros((n,n))
jaccard.fill(-1)

for i in range(n):
    for j in range(i+1,n):
        union = len(A[i])
        intersection = 0
        for k in A[j]:
            if k in A[i]:
                intersection+=1
            else:
                union +=1
        jaccard[i][j]=intersection/union
        
#Using Complete Linkage Strategy
        
mat = np.copy(jaccard)
        
hierarchy_complete = []
B=[]
for i in range(n):
    B.append([i])
    
hierarchy_complete.append(B)

while(np.amax(mat)!=-1):
    temp = np.where(mat == np.amax(mat))
    locations = list(zip(temp[0],temp[1]))
    last= hierarchy_complete[-1].copy()
    for i in locations:
        for j in last:
            if i[0] in j:
                part1=j.copy()
            if i[1] in j:
                part2=j.copy()
                
        if(part1!=part2):
            x=0
            for j in part1:
                for k in part2:
                    if j<k:
                        if(mat[j][k]!=-1):                    
                            if(j!=i[0] or k!=i[1]):
                                x+=1                        
                    else:
                        if(mat[k][j]!=-1):
                            x+=1
            mat[i[0]][i[1]]=-1
            if(x==0):
                last.remove(part1)
                last.remove(part2)
                part1.extend(part2)
                last.append(part1)
    hierarchy_complete.append(last)
    
#Making 9 clusters for Complete Linkage Strategy
    
cl = hierarchy_complete[-2].copy()  #Similarity reaches zero mutually at this stage
index = 0                           #Hence, clusters can be combined randomnly to form 9 clusters
while(len(cl)>9):
    cl[index%len(cl)].extend(cl[(index+1)%len(cl)])
    del cl[(index+1)%len(cl)]
    index+=1                   
    
#Using Single Linkage Strategy
        
mat = np.copy(jaccard)
        
hierarchy_single = []
B=[]
for i in range(n):
    B.append([i])
    
hierarchy_single.append(B)

while(np.amax(mat)!=-1):
    temp = np.where(mat == np.amax(mat))
    locations = list(zip(temp[0],temp[1]))
    last= hierarchy_single[-1].copy()

    for i in locations:
        for j in last:
            if i[0] in j:
                part1=j
            if i[1] in j:
                part2=j
        for j in part1:
            for k in part2:
                if j<k:
                    mat[j][k]=-1
                else:
                    mat[k][j]=-1
        last.remove(part1)
        try:
            last.remove(part2)
        except ValueError:
            pass
        
        last.append(list(set(part1)|set(part2)))
    
    hierarchy_single.append(last)
    
#Making 9 clusters for Single Linkage Strategy
    
mat = np.copy(jaccard)
        
hie = []
B=[]
for i in range(n):
    B.append([i])
    
hie.append(B)

while(np.amax(mat)!=-1):
    temp = np.where(mat == np.amax(mat))
    locations = list(zip(temp[0],temp[1]))
    last= hie[-1].copy()

    for i in locations:
        if(len(last)>9):
            for j in last:
                if i[0] in j:
                    part1=j
                if i[1] in j:
                    part2=j
            for j in part1:
                for k in part2:
                    if j<k:
                        mat[j][k]=-1
                    else:
                        mat[k][j]=-1
            last.remove(part1)
            try:
                last.remove(part2)
            except ValueError:
                pass
            last.append(list(set(part1)|set(part2)))
    
    hie.append(last)
    
    if(len(last)==9):
        break

sl=hie[-1]




    
    
        
    






        
        
        
    


