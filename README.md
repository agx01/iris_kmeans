# K-Means Clustering on Iris Dataset

## Problem Statement
The use of iris data set for the prediction of species is a classic example for classification problem.
We will be implementing K-means clustering algorithm on this dataset and validate the accuracy of our model using the actual species data.

## Strategy
The strategy used for K-Means is to initialize centroids using the first 3 records of the X values. Then we calculate the distance from each point to the centroid and mark that record with the closest centroid value. Then re-create the centroid by finding the mean of each cluster. Then re calculate the distances and re- create the clusters based on the closest distances. The distances changes will provide the error value, if the error value goes to zero then the clusters have not changed. This becomes the final the clusters.
We repeat this process with different metrics to calculate the distances from the centroids to sample points. For this project, I am using:
1.	Euclidean Distance
2.	Manhattan Distance

## Folders:
**data** - Iris dataset is stored in the folder

## Choosing the right K- values
For our experiment, we know to use 3 clusters because of the number of classes available in the dataset.
However, in actual scenario, we will not be informed of the groups available data.
We use the Elbow method to choose the number of clusters in the data.

For this we run the sklearn Kmeans algorithm, and then measure WCSS value across the number of clusters picked.
![The Elbow Method](https://github.com/agx01/iris_kmeans/blob/main/ElbowMethod.png?raw=true)


## K-Means Clustering
K-Means clustering algorithms is used to find natural groups in the data.
The training method for K-means is very simple as it only stores the data. However, the predict method is compute intensive as it calculates the distances between the points and centroids multiple times.

Main challenges of K-means algorithms:
1. Picking the right centroids
2. Picking the right number of clusters

## Results

### Euclidean Distance metrics

Using the Euclidean distance metric, we get the following results. Plotting the box-plots for the actual classes and predicted clusters, gives a relation between the cluster and the label to use as mapping.
| Species | Cluster Number |
| :---: | :---: |
| Iris-Setosa | Cluster 2 |
| Iris-virginica | Cluster 0 |
| Iris-versicolor | Cluster 1 |

![Box Plot of Features (Euclidean Distance)](https://github.com/agx01/iris_kmeans/blob/main/EuclideanDistance.png?raw=true)
    
For this metric, we get an accuracy of **86.667%**.
![Results of Euclidean Metric](https://github.com/agx01/iris_kmeans/blob/main/Euclidean_results.png?raw=true)
    
### Manhattan Distance metrics

Using the Manhattan distance metric, we get the following results. Plotting the box-plots for the actual classes and predicted clusters, gives a relation between the cluster and the label to use as mapping.
| Species | Cluster Number |
| :---: | :---: |
| Iris-Setosa | Cluster 2 |
| Iris-virginica | Cluster 0 |
| Iris-versicolor | Cluster 1 |

![Box Plot of Features (Manhattan Distance)](https://github.com/agx01/iris_kmeans/blob/main/ManhattanDistance.png?raw=true)
    
For this metric, we get an accuracy of **86.667%**.
![Results of Euclidean Metric](https://github.com/agx01/iris_kmeans/blob/main/Manhattan_results.png?raw=true)