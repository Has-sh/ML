import sys
sys.path.append("C:/Users/hasha/Desktop/Assignment 2 Data and Source Codes")
import numpy as np
from Distance import Distance
class KMemoids:
    def __init__(self, dataset, K=2, distance_metric="cosine"):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        self.distance_metric = distance_metric
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_medoids stores the cluster medoid for each cluster in a dictionary
        # # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.cluster_medoids = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class

    def calculateLoss(self):
        # Calculate the total distance within clusters based on the chosen distance metric
        L = 0.0
        for cluster_point in self.clusters.items():
            for datapoint in cluster_point[1]:
                L += np.sum(Distance.calculateCosineDistance(datapoint,self.cluster_medoids[cluster_point[0]]))
        return L
    
    def run(self):
        # Initialization of variables for tracking losses and convergence
        losses = []
        prev_losses = [float('inf')] * 5
        prev_loss = float('inf')
        flag = 0
        tolerance = 1e-8

        #initilization of medoids
        for i in range(self.K):
            random_index = np.random.choice(len(self.dataset), size=1)[0]
            self.cluster_medoids[i] = self.dataset[random_index]
        
        # Assignment of data points to the nearest medoid
        for datapoint in self.dataset:
                distance_prev = float('inf')
                closest_medoid = None
                for medoid_index in range(self.K):
                    new_distance = Distance.calculateCosineDistance(datapoint, self.cluster_medoids[medoid_index])
                    if new_distance < distance_prev:
                        distance_prev = new_distance
                        closest_medoid = medoid_index

                self.clusters[closest_medoid].append(datapoint)
                
        # Iterative update of medoids to minimize total distance within clusters
        for i in range(self.K):
            if self.clusters[i]:
                min_total_distance = float('inf')
                new_medoid = None

                for medoid_candidate in self.clusters[i]:
                    total_distance = 0

                    for other_point in self.clusters[i]:
                        distance = Distance.calculateCosineDistance(medoid_candidate, other_point)
                        total_distance += distance
                    
                    if total_distance < min_total_distance:
                        min_total_distance = total_distance
                        new_medoid = medoid_candidate
                    
                self.cluster_medoids[i] = new_medoid
            
        return self.cluster_medoids, self.clusters, self.calculateLoss()