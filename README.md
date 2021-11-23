# K-Means Clustering on Iris Dataset

## Problem Statement
The use of iris data set for the prediction of species is a classic example for classification problem.
We will be implementing K-means clustering algorithm on this dataset and validate the accuracy of our model using the actual species data.

## Strategy
The strategy we will be implementing is a classic implementation of the K-means clustering algorithm.
We will use the exact number of clusters as the number of classes in the dataset to validate the accuracy of various metrics on the K-means.

## Folders:
**data** - Iris dataset is stored in the folder

## Choosing the right K- values
For our experiment, we know to use 3 clusters because of the number of classes available in the dataset.
However, in actual scenario, we will not be informed of the groups available data.
We use the Elbow method to choose the number of clusters in the data.

For this we run the sklearn Kmeans algorithm, and then measure WCSS value across the number of clusters picked.
![The Elbow Method](https://github.com/agx01/iris_kmeans/blob/main/ElbowMethod.png?raw=true)


## 