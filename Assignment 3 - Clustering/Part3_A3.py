"""
Assignment 3: Clustering
Part 3

@author: Divyam Jain
         17HS20047
"""
#Importing some essential libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Importing the dataset

dataset = pd.read_csv("AAAI.csv")

#Importing the clusters

from Part1_A3 import cl
from Part1_A3 import sl
from Part2_A3 import required_cluster

gold = dataset.iloc[:,3]

def NMI(cluster, gold):
    
    n=len(gold)
    table=pd.DataFrame(0,index=range(0,len(cluster)) ,columns=pd.unique(gold))
    for i in cluster:
        for j in i:
            table[gold[j]][cluster.index(i)]+=1
    a=0
    for i in table.index:
        for j in table.columns:
            sum0=table.loc[i,:].sum()
            sum1=table.loc[:,j].sum()
            if ((n*table[j][i])/(sum0*sum1))!=0:
                a = a + table[j][i]/n*math.log((n*table[j][i])/(sum0*sum1))
                
    b=0
    column_sum=table.sum(axis=0)
    for i in column_sum:
        if i!=0:
            b=b-i/n*math.log(i/n)
    
    c=0
    row_sum=table.sum(axis=1)
    for i in row_sum:
        if i!=0:
            c=c-i/n*math.log(i/n)
            
    return (2*a/(b+c))

NMI_complete = NMI(cl,gold)
NMI_single = NMI(sl,gold)
NMI_graph = NMI(required_cluster,gold)            
    
            
    
    
    


