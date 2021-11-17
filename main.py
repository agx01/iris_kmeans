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

from sklearn.cluster import KMeans as kms


class K_Means_Cluster:
    
    def __init__(self, num_clusters):
        data = pd.read_csv('data/iris.data', names = ['slength', 'swidth', 'plength', 'pwidth', 'species'])
        c1 = data['slength'].values    
        c2 = data['swidth'].values
        c3 = data['plength'].values
        c4 = data['pwidth'].values
        
        self.initial_visualization(data)
        
        #Input array
        self.X = np.array(list(zip(c1,c2,c3,c4)), dtype=np.float32)
        
        #Number of clusters
        self.num_clusters = num_clusters
        
        data["species"] = data["species"].astype('category')
        data["species_cat"] = data["species"].cat.codes
        
        #Label data
        self.Y = np.array(data['species_cat'].values)
        
        self.predict_all()
        
    def check_accuracy(self, data, metric):
        iris_setosa = data[data["species"] == "Iris-setosa"]
        iris_virginica = data[data["species"] == "Iris-virginica"]
        iris_versicolor = data[data["species"] == "Iris-versicolor"]
        
        if metric == "Euclidean":
            pred_setosa = data[data["pred_Y"] == 2]
            pred_virginica = data[data["pred_Y"] == 1]
            pred_versicolor = data[data["pred_Y"] == 0]
            
            #True values
            true_setosa_count = len(iris_setosa[iris_setosa["pred_Y"] == 2])
            true_virginica_count = len(iris_virginica[iris_virginica["pred_Y"] == 0])
            true_versicolor_count = len(iris_versicolor[iris_versicolor["pred_Y"] == 1])
        else:
            pred_setosa = data[data["pred_Y"] == 2]
            pred_virginica = data[data["pred_Y"] == 1]
            pred_versicolor = data[data["pred_Y"] == 0]
            
            #True values
            true_setosa_count = len(iris_setosa[iris_setosa["pred_Y"] == 2])
            true_virginica_count = len(iris_virginica[iris_virginica["pred_Y"] == 1])
            true_versicolor_count = len(iris_versicolor[iris_versicolor["pred_Y"] == 0])
        
        #Counting the actual values
        act_setosa_count = len(iris_setosa)
        act_virginica_count = len(iris_virginica)
        act_versicolor_count = len(iris_versicolor)
        
        #Counting the predicted values
        pred_setosa_count = len(pred_setosa)
        pred_virginica_count = len(pred_virginica)
        pred_versicolor_count = len(pred_versicolor)
        
        #False values
        false_setosa_count = act_setosa_count - true_setosa_count
        false_virginica_count = act_virginica_count - true_virginica_count
        false_versicolor_count = act_versicolor_count - true_versicolor_count
        
        total_correct_count = true_setosa_count + true_virginica_count + true_versicolor_count
        total_records = len(data)
        
        print(f"Overall Accuracy using {metric} distance is : {(total_correct_count/total_records)*100}")
        
    def initial_visualization(self, data):
        print("Understanding the data:")
        print(data.info())
        print(data.describe())
        
        #Choosing the number of clusters
        #Within cluster sum of squares
        wcss = []
        x = data.iloc[:, [0, 1 ,2, 3]].values        
        for i in range(1, 11):
            kmeans = kms(n_clusters=i, init='k-means++', max_iter=300, n_init = 10, random_state=0)
            kmeans.fit(x)
            wcss.append(kmeans.inertia_)
            
        plt.plot(range(1,11), wcss)
        plt.title('The Elbow Methdod')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()
        
        #Visualizing the data in 3D
        kmeans = kms(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(x)
        fig = plt.figure(figsize = (15, 15))
        ax = fig.add_subplot(111, projection='3d')
        plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans ==0, 1], s =100, c = 'purple', label = 'Iris-setosa')
        plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans ==1, 1], s =100, c = 'orange', label = 'Iris-versicolor')
        plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans ==2, 1], s =100, c = 'green', label = 'Iris-virginica')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =100, c = 'red', label = 'Centroids')
        plt.suptitle("Sample and Centroid Visualization")
        plt.show()
        
        print("Visualization complete")
    
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
        #print(f"Accuracy Score (Euclidean distance): {accuracy_score(self.Y, clusters)}")
        self.data_visualization(clusters, "Euclidean")
        
        
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
        
        self.data_visualization(clusters, "Manhattan")
        
    def data_visualization(self, clusters, metric):
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
        
        """
        cluster_setosa = iris_outcome3["species" == "Iris-setosa"]
        cluster_versicolor = iris_outcome3["species" == "Iris-versicolor"]
        cluster_virginica = iris_outcome3["species" == "Iris-virginica"]
        cluster_setosa = cluster_setosa.count >= cluster_setosa.count
        cluster_versicolor = cluster_versicolor.count >= cluster_versicolor.count
        cluster_virginica = cluster_virginica.count >= cluster_virginica.count
        """
        
        fig2, axes2 = plt.subplots(2,4, sharey=True)
        #fig2.set_title(f"Actual vs Predicted (metric)")
        plt.suptitle(f"Actual vs Predicted ({metric})")
        
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
        
        self.check_accuracy(df2, metric)
        
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
    k_means = K_Means_Cluster(num_clusters=3)