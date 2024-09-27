import sys
sys.path.append("C:/Users/hasha/Desktop/Assignment 2 Data and Source Codes")
from Distance import Distance
import numpy as np
import random
class KMeans:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.previous_cluster_centers = {i: np.zeros_like(self.dataset[0]) for i in range(self.K)}    
    
    def calculateLoss(self):
        # Calculate the sum of squared distances between data points and their assigned cluster centers
        L = 0.0
        for cluster_point in self.clusters.items():
            for datapoint in cluster_point[1]:
                L += np.sum(np.linalg.norm(np.array(datapoint) - np.array(self.cluster_centers[cluster_point[0]]))**2)
        return L

    def run(self):
        # Initialize variables for tracking losses and convergence
        losses=[]
        prev_losses = [float('inf')]*5
        prev_loss=float('inf')
        flag=0
        tolerance = 1e-8

        # Initialize cluster centers by randomly selecting data points
        for i in range(self.K):
                random_index = np.random.choice(len(self.dataset), size=1)[0]
                self.cluster_centers[i] = self.dataset[random_index]

        # Continue iterations until convergence                
        while not all(np.all(np.abs(np.array(self.previous_cluster_centers[i]) - np.array(self.cluster_centers[i])) < tolerance)for i in range(self.K)):
            
            # Update the previous cluster centers                        
            self.previous_cluster_centers = self.cluster_centers.copy()
            # Clear current clusters
            self.clusters = {i: [] for i in range(self.K)}

            # Assign each data point to the closest cluster
            for datapoint in self.dataset:
                distancePrev = float('inf')
                closest_cluster = None
                for cluster_index in range(self.K):
                    newdistance = Distance.calculateMinkowskiDistance(datapoint,self.cluster_centers[cluster_index],2)#distance fucntion idk which to use
                    if newdistance < distancePrev:
                        distancePrev = newdistance
                        closest_cluster = cluster_index

                self.clusters[closest_cluster].append(datapoint)

            # Update cluster centers based on the mean of data points in each cluster
            for i in range(self.K):
                if self.clusters[i]:
                    self.cluster_centers[i] = np.mean(np.array(self.clusters[i]), axis=0)
                    
        # Return the final cluster centers, clusters, and the calculated loss
        return self.cluster_centers, self.clusters, self.calculateLoss()
