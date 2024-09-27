import numpy as np
class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        # Calculate distances between the input instance and all data points in the dataset
        distances = []
        for i, data in enumerate(self.dataset):
                # Use the specified similarity function to calculate distance
                if self.similarity_function_parameters is not None:
                    distance = self.similarity_function(instance, data, **self.similarity_function_parameters)
                else:
                    distance = self.similarity_function(instance, data)
                distances.append((distance, self.dataset_label[i], i))
        
        # Sort distances and select the K nearest neighbors
        distances.sort(key=lambda x: x[0])

        # Extract labels of the K nearest neighbors
        closest=distances[:self.K]

        closest_labels = []
        for _, label, _ in closest:
            closest_labels.append(label)

        #find the most common label among the K nearest neighbors
        unique_label, count = np.unique(closest_labels, return_counts=True)
        closest_label = unique_label[np.argmax(count)]

        return closest_label
    




