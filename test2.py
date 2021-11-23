# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 01:04:26 2021

@author: Arijit Ganguly
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 

iris = load_iris()
data = iris.data[:,:2]

iris2 = pd.read_csv("data/iris.data").to_numpy()
data = iris2[:, :2]

plt.scatter(data[:,0],data[:,1],c='black')
plt.show()

k = 3
max_iteration = 2

def Distance2Point(point1, point2):
    dis = sum((point1 - point2)**2)**0.5
    return dis

def KMean(data):

    centroids = {}

    for i in range(k):
        centroids[i] = data[i]

    classes = {}
    for iteration in range(max_iteration):
        classes = {}
        for classKey in range(k):
            classes[classKey] = []

        for dataPoint in data: 
            Distance = []
            for centroid in centroids:
                dis = Distance2Point(dataPoint, centroids[centroid])
                Distance.append(dis)

            minDis = min(Distance)
            minDisIndex = Distance.index(minDis)
            classes[minDisIndex].append(dataPoint)
           
        oldCentroid = dict(centroids)
        
        for classKey in classes:
            classData = classes[classKey]
            NewCentroid = np.mean(classData, axis = 0)
            centroids[classKey] = NewCentroid
        
        isFine = True
        for centroid in oldCentroid:
            oldCent = oldCentroid[centroid]
            curr = centroids[centroid]
            
        if np.sum((curr - oldCent)/oldCent * 100) > 0.001:
            isFine = False

        if isFine:
            break
    return centroids, classes

data = iris2[:, :4]
centroids, classes = KMean(data[:, :4])

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
