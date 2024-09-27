import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value=0.0
        # Counting occurrences of unique labels
        _, label_counts=np.unique(labels, return_counts=True)
        total_instances=len(labels) # Total number of instances in the dataset
        # Calculate entropy using the formula: -sum(p * log2(p)) for each unique label's count
        for count in label_counts:
            prob=count/total_instances # Probability of a label occurrence
            entropy_value-=prob*np.log2(prob) # Entropy calculation for each label

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy=0.0
        unique_values=np.unique(dataset[:, attribute]) # Unique values for the specified attribute
        
        for value in unique_values:
            # Finding indices where attribute equals the value
            subset_indices=np.where(dataset[:, attribute] == value)[0]
            # Extract labels corresponding to these indices
            subset_labels=[labels[i] for i in subset_indices]
            # Calculating the weight based on the total number of labels
            subset_weight=len(subset_labels)/len(labels)
            # Calculating entropy for the subset
            entropy_value=self.calculate_entropy__(dataset,subset_labels)
            # Calculating avg entropy for each subset
            average_entropy+=subset_weight*entropy_value

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain=0.0
        
        # Calculating the entropy of the original dataset
        original=self.calculate_entropy__(dataset, labels)
        # Calculating the average entropy
        average=self.calculate_average_entropy__(dataset, labels, attribute)
        # Computing the information gain by subtracting the average entropy from the original entropy
        information_gain=original-average

        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0

        # Calculating unique values
        unique_values, value_counts=np.unique(dataset[:, attribute],return_counts=True)
        total_instances = len(labels) # Total number of instances in the dataset
        # Calculating using the formula: -sum(p * log2(p))
        for count in value_counts:
            prob=count/total_instances # Probability of a value occurrence
            intrinsic_info-=prob*np.log2(prob) # Intrinsic information calculation for each value

        return intrinsic_info
    
    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        # Calculating information gain and intrinsic information
        information_gain=self.calculate_information_gain__(dataset, labels, attribute)
        intrinsic_info=self.calculate_intrinsic_information__(dataset, labels, attribute)
        # Checking if intrinsic information is zero to avoid division by zero had some issues prev
        if intrinsic_info==0:
            return 0.0 # If intrinsic information is zero, return 0 as gain ratio
        
        gain_ratio=information_gain/intrinsic_info  # gain ratio
        return gain_ratio

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        # Retrieving unique labels
        unique_labels,label_counts=np.unique(labels, return_counts=True)
        # If all instances have the same label, return a leaf node with that label
        if len(unique_labels)==1:
                return TreeLeafNode(dataset,unique_labels[0])
        
        # If all attributes are used or dataset is empty, return a leaf node with the majority label
        if len(used_attributes)==len(self.features) or len(dataset) == 0:
            majority_label = unique_labels[np.argmax(label_counts)]
            return TreeLeafNode(dataset, majority_label)
        
        best_attribute=None
        best_information_gain=-np.inf
        dataset=np.array(dataset)

        # Iterate through each attribute to find the one with the best information gain
        for attribute in range(len(self.features)):
                if attribute not in used_attributes:
                    information_gain=self.calculate_information_gain__(dataset, labels, attribute)
                    if information_gain>best_information_gain:
                        best_attribute=attribute
                        best_information_gain=information_gain

        # If no attribute provides information gain return a leaf node with the majority label
        if best_attribute is None:
            majority_label=unique_labels[np.argmax(label_counts)]
            return TreeLeafNode(dataset, majority_label)
        
        # Creating a non-leaf node using the best attribute found
        node=TreeNode(self.features[best_attribute])
        used_attributes.append(best_attribute)
        unique_values=np.unique(dataset[:, best_attribute])

        # Creating subtrees for each unique value of the best attribute 
        for value in unique_values:
            subset_indices=np.where(dataset[:, best_attribute]==value)[0]
            subset_data=dataset[subset_indices]
            subset_labels=[labels[i] for i in subset_indices]
            sub_node=self.ID3__(subset_data,subset_labels,used_attributes.copy())
            node.subtrees[value]=sub_node
        
        return node

                        
    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label=None
        node=self.root
        
        # Traverse the decision tree to predict the label
        while True:
            if isinstance(node, TreeLeafNode):
                predicted_label=node.labels
                break # If a leaf node is reached assign its label as the predicted label

            attribute_value=x[self.features.index(node.attribute)]
            if attribute_value in node.subtrees:
                node=node.subtrees[attribute_value]
            else:
                # If attribute value not found in subtrees determine majority label in the node
                labels_in_node=[sub_node.labels for sub_node in node.subtrees.values()]
                predicted_label=max(set(labels_in_node), key=labels_in_node.count)
                break # Break the loop as the prediction is made

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")