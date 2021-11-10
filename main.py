# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:19:08 2021

@author: Arijit Ganguly
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import random

class KMeans:
    
    def __init__(self, num_clusters):
        data = pd.read_csv('data/iris.data', names = ['slength', 'swidth', 'plength', 'pwidth', 'species'])
        c1 = data['slength'].values    
        c2 = data['swidth'].values
        c3 = data['plength'].values
        c4 = data['pwidth'].values
        
        #Input array
        self.X = np.array(list(zip(c1,c2,c3,c4)), dtype=np.float32)
        
        #Number of clusters
        self.num_clusters = num_clusters
        
        self.predict()
    
    def predict(self):
        X = self.X
        
        #Assigning the centroids
        c1 = [X[0][0],X[1][0],X[2][0]] 
        #first feature cluster centroids
        c2 = [X[0][1],X[1][1],X[2][1]] 
        #second feature cluster centroids
        c3 = [X[0][2],X[1][2],X[2][2]] 
        #third feature cluster centroids
        c4 = [X[0][3],X[1][3],X[2][3]] 
        #fourth feature cluster centroids
        
        c = np.array(list(zip(c1, c2, c3, c4)))
        
        print(c)
        
        #Initlizing a variable to store the old centroids
        c_old = np.zeros(c.shape)
        
        #Stores the centroid of the nearest point
        clusters = np.zeros(len(X))
        
        #Stores the error or difference between old centroid and new centroids
        error = self.dist(c, c_old, None)
        
        
        while error != 0:
            #Assigning the nearest point to its cluster
            for i in range(len(X)):
                distances = self.dist(X[i], c)
                cluster = np.argmin(distances)
                clusters[i] = cluster
                
            #Push the centroid values to the old centroid values
            c_old = deepcopy(c)
            
            #Finding the new mean of each cluster
            for i in range(self.num_clusters):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                c[i] = np.mean(points, axis=0)
            error = self.dist(c, c_old, None)
            
        print(c)
        
        print(clusters)
        
        print(error)

    # Returns a vector norm of cluster centroids and the corresponding features
    def dist(self, a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)
    
    
        
if __name__ == "__main__":
    k_means = KMeans(num_clusters=3)