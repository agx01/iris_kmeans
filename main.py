# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:19:08 2021

@author: Arijit Ganguly
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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
        
        data["species"] = data["species"].astype('category')
        data["species_cat"] = data["species"].cat.codes
        
        #Label data
        self.Y = np.array(data['species_cat'].values)
        
        self.predict_all()
    
    def predict_all(self):
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
        
        #Initlizing a variable to store the old centroids
        c_old = np.zeros(c.shape)
        
        #Stores the centroid of the nearest point
        clusters = np.zeros(len(X))
        
        #Stores the error or difference between old centroid and new centroids
        error = self.dist(c, c_old, None)
        
        #Calculating using the Euclidean distance
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
        
        print(f"Y labels: {self.Y}")
        print(f"Predicted Values: {clusters}")
        print(f"Accuracy Score (Euclidean distance): {accuracy_score(self.Y, clusters)}")
        self.data_visualization(clusters)
        
        
        #Calculating using the Manhattan distance
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
        
        #Initlizing a variable to store the old centroids
        c_old = np.zeros(c.shape)
        
        #Stores the centroid of the nearest point
        clusters = np.zeros(len(X))
        
        #Stores the error or difference between old centroid and new centroids
        error = self.dist(c, c_old, None)
        
        #Calculating using the Euclidean distance
        while error != 0:
            #Assigning the nearest point to its cluster
            for i in range(len(X)):
                distances = self.manhattan_dist(X[i], c)
                cluster = np.argmin(distances)
                clusters[i] = cluster
                
            #Push the centroid values to the old centroid values
            c_old = deepcopy(c)
            
            #Finding the new mean of each cluster
            for i in range(self.num_clusters):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                c[i] = np.mean(points, axis=0)
            error = self.dist(c, c_old, None)
        
        print(f"Y labels: {self.Y}")
        print(f"Predicted Values: {clusters}")
        print(f"Accuracy Score (Manhattan distance): {accuracy_score(self.Y, clusters)}")
        self.data_visualization(clusters)
        
    def data_visualization(self, clusters):
        df2 = pd.read_csv("data/iris.data", names = ['slength',
                                                     'swidth',
                                                     'plength',
                                                     'pwidth',
                                                     'species'])
        df2["pred_Y"] = clusters
        iris_outcome = pd.crosstab(df2['species'], "count")
        print(f"Actual Class Count: {iris_outcome}")
        iris_outcome2 = pd.crosstab(df2['pred_Y'], "count")
        print(f"Predicted Class Count: {iris_outcome2}")
        iris_outcome3 = pd.crosstab([df2['species'],df2['pred_Y']], "count")
        print(f"Predicted Class Count: {iris_outcome3}")
        
        fig2, axes2 = plt.subplots(2,4, sharex=True, sharey=True)
        
        #Actual data charts
        sns.barplot(ax=axes2[0][0], x = "species", y = "slength", data=df2)
        sns.barplot(ax=axes2[0][1], x = "species", y = "swidth", data=df2)
        sns.barplot(ax=axes2[0][2], x = "species", y = "plength", data=df2)
        sns.barplot(ax=axes2[0][3], x = "species", y = "pwidth", data=df2)
        
        #Predicted data charts
        sns.barplot(ax=axes2[1][0], x = "pred_Y", y = "slength", data=df2)
        sns.barplot(ax=axes2[1][1], x = "pred_Y", y = "swidth", data=df2)
        sns.barplot(ax=axes2[1][2], x = "pred_Y", y = "plength", data=df2)
        sns.barplot(ax=axes2[1][3], x = "pred_Y", y = "pwidth", data=df2)
        
        plt.show()
        
        """
        new_df = pd.DataFrame(self.X, columns = ['slength', 'swidth', 'plength', 'pwidth'])
        new_df['actual_Y'] = self.Y
        new_df['pred_Y'] = clusters
        """
        """
        actualY_df = new_df.loc[:, new_df.columns != 'pred_Y']
        
        print(new_df.head())
        sns.set_style("whitegrid")
        sns.pairplot(actualY_df, vars=['slength'], hue="actual_Y", palette="tab10")
        
        predY_df = new_df.loc[:, new_df.columns != 'actual_Y']
        sns.set_style("whitegrid")
        sns.PairGrid(predY_df, vars = ['slength'], hue="pred_Y", palette="tab10")
        """
        
        """
        #using the cross tab
        iris_outcome = pd.crosstab(index=new_df["actual_Y"])
        
        # using a 3-D scatter plot
        fig, axs =  plt.subplots(1, 1)
        axs[0] = Axes3D(fig)
        axs[0].scatter(new_df['slength'].values, 
                   new_df['swidth'].values, 
                   new_df['plength'], 
                   s=new_df['pwidth']+10, 
                   c=new_df['actual_Y'],
                   cmap = "tab10")
        axs[0].set_xlabel("Sepal Length")
        axs[0].set_ylabel("Sepal Width")
        axs[0].set_zlabel("Petal Length")
        
        plt.show()
        
        """
        
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
                print("X[i]:")
                print(X[i])
                print("c:")
                print(c)
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

    # Returns a vector norm of cluster centroids and the corresponding features
    def dist(self, a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)
    
    #Calculate the euclidean distance
    def euclid_dist(self, a, b):
        return np.sum(np.square(a-b))
    
    #Calculate the manhattan distance
    def manhattan_dist(self, a, b):
        return np.abs(a - b).sum(-1)
    
        
if __name__ == "__main__":
    k_means = KMeans(num_clusters=3)