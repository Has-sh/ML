import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Calculate the counts for each merge in the hierarchical clustering
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create the linkage matrix for the dendrogram
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the dendrogram
    dendrogram(linkage_matrix, **kwargs)



# Calculate average distance within the cluster for a given data point
def alpha(datapoint, cluster):
    cluster_size = len(cluster)
    distances = 0.0

    for i in range(cluster_size):
        if not np.array_equal(datapoint, cluster[i]):
            distances += np.linalg.norm(datapoint - cluster[i])
    if (cluster_size - 1) > 0: 
        average_distance = distances / (cluster_size - 1) 
    else: 
        average_distance= 0
    return average_distance

# Calculate average distance to the nearest cluster for a given data point
def beta(datapoint, clusters):
    min_distance = float('inf')

    for cluster in clusters:
        if cluster.any():
            distances_to_cluster = []
            for clusterpoint in cluster:
                distance_to_point = np.linalg.norm(datapoint - clusterpoint)
                distances_to_cluster.append(distance_to_point)

            mean_distance_to_cluster = np.mean(distances_to_cluster)
            
            min_distance = min(min_distance, mean_distance_to_cluster)

    return min_distance

# Calculate silhouette score for a given data point in a cluster
def silhouetteScore(datapoint, cluster, clusters):
    if(len(cluster)<=1):
        return 0
    
    a = alpha(datapoint, cluster)
    b = beta(datapoint, clusters)
    return (b - a) / max(a, b)

# Calculate the average silhouette value for a cluster
def average_silhouette_value(cluster, clusters):
    silhouette_scores = []
    for datapoint in cluster:
        datapoint_silhouette = silhouetteScore(datapoint, cluster, clusters)
        silhouette_scores.append(datapoint_silhouette)

    average_silhouette = np.mean(silhouette_scores)

    return average_silhouette

dataset = pickle.load(open("data/part3_dataset.data", "rb"))

linkages = ['single', 'complete']
distances = ['euclidean', 'cosine']
K = [2, 3, 4, 5]



for distance in distances:
    for linkage in linkages:
        hac = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage=linkage, metric=distance, compute_distances=True)
        hac.fit(dataset)
        
        # Get the actual number of clusters formed
        cluster_labels = hac.fit_predict(dataset)
        k = len(np.unique(cluster_labels))

        # Plot Dendrogram
        plt.figure(figsize=(10, 5))
        plt.title(f'Dendrogram - Linkage: {linkage}, Metric: {distance}')
        plot_dendrogram(hac, truncate_mode='level', p=15)
        plt.xlabel('Index')
        plt.ylabel('Distance')
        plt.show()
        
        # Perform Silhouette calc
        clusters = [dataset[cluster_labels == i] for i in range(k)]
        
        silhouette_scores = []
        for i, datapoint in enumerate(dataset):
            datapoint_silhouette = silhouetteScore(datapoint, clusters[cluster_labels[i]], clusters)
            silhouette_scores.append(datapoint_silhouette)

        # Plot Silhouette Values
        plt.figure(figsize=(10, 5))
        plt.title(f'Silhouette Plot - Linkage: {linkage}, Metric: {distance}, K: {k}')
        plt.bar(range(len(dataset)), silhouette_scores, color='blue')
        plt.axhline(y=np.mean(silhouette_scores), color="red", linestyle="--")
        plt.xlabel('Data Point')
        plt.ylabel('Silhouette Value')
        plt.show()

        # Calculate the average silhouette value for each cluster
        avg_silhouettes_per_cluster = [average_silhouette_value(cluster, clusters) for cluster in clusters]

        print(f'Average Silhouette Scores per Cluster - Linkage: {linkage}, Metric: {distance}, K: {k}:')

        for i, avg_silhouette_cluster in enumerate(avg_silhouettes_per_cluster):
            print(f'  Cluster {i + 1}: {avg_silhouette_cluster}')