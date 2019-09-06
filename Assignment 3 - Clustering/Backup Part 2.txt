"""
Assignment 3: Clustering
Part 2

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
        
#Creating the graph
        
import networkx as nx

threshold = 0.1

G=nx.Graph()
H=nx.path_graph(n)
G.add_nodes_from(H)

edges=[]

for i in range(n):
    for j in range(i+1,n):
        if(jaccard[i][j]>threshold):
            edges.append((i,j))

G.add_edges_from(edges)

#Applying the Girvan-Newman Clustering Algorithm

graph = G.copy()
clustering=[G]             
while(nx.number_of_edges(graph)>0):
    centralities = nx.edge_betweenness_centrality(graph)
    x=list(centralities)[list(centralities.values()).index(max(list(centralities.values())))]
    copy=graph.copy()
    copy.remove_edge(*x)
    clustering.append(copy)
    graph.remove_edge(*x)
    
    
#Finding 9 clusters
    
number_of_clusters = 9
    
for i in clustering:
    if(nx.number_connected_components(i)==number_of_clusters):
        break
required_cluster = list(nx.connected_components(i))
    

    







