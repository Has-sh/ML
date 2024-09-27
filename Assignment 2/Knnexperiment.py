import sys
sys.path.append("C:/Users/hasha/Desktop/Assignment 2 Data and Source Codes")
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Distance import Distance
from Part1.Knn import KNN

# Function to calculate confidence interval
def calculate_confidence_interval(min_loss):
    mean_val = np.mean(min_loss)
    margin_of_error = np.std(min_loss) * 1.96 / np.sqrt(len(min_loss))
    return mean_val, mean_val - margin_of_error, mean_val + margin_of_error


dataset, labels = pickle.load(open("data/part1_dataset.data", "rb"))

# Initializing variables to keep track of the best hyperparameters and accuracy
best_accuracy = 0.0
best_hyperparameters = {}

# Hyperparameter values to be tested
K_values = [1, 5, 10, 15, 30, 50]
similarity_functions = [Distance.calculateCosineDistance, Distance.calculateMahalanobisDistance, Distance.calculateMinkowskiDistance]

# Creating Stratified K-Fold cross-validator with 10 splits
kfold = StratifiedKFold(n_splits=10)
list_avg_accuracy=[]
for k in K_values:
    for similarity_function in similarity_functions:
        accuracies = []
        
        # Performing 5 iterations for each set of hyperparameters
        for _ in range(5):
            for train_indices, test_indices in kfold.split(dataset, labels):
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)

                # Splitting the data into training and testing sets
                train_data, test_data = dataset[train_indices], dataset[test_indices]
                train_labels, test_labels = labels[train_indices], labels[test_indices]

                # Creating KNN instance based on the similarity function
                if similarity_function == Distance.calculateMahalanobisDistance:
                    S_minus_1 = np.linalg.inv(np.cov(dataset, rowvar=False))
                    knn = KNN(train_data, train_labels, similarity_function, {'S_minus_1': S_minus_1}, K=k)
                elif similarity_function == Distance.calculateMinkowskiDistance:
                    knn = KNN(train_data, train_labels, similarity_function, {'p': 2}, K=k)
                else:
                    knn = KNN(train_data, train_labels, similarity_function, K=k)

                # Counting correct predictions for each test instance
                correct_predictions = 0
                for i in range(len(test_data)):
                    prediction = knn.predict(test_data[i])
                    if prediction == test_labels[i]:
                        correct_predictions += 1

                # Calculating accuracy for the current fold
                accuracy = correct_predictions / len(test_data)
                accuracies.append(accuracy)

            # Calculating average accuracy for the current set of hyperparameters
            avg_accuracy = np.mean(accuracies)
            list_avg_accuracy.append(avg_accuracy)

        # Calculating overall average accuracy for the current set of hyperparameters
        confidence_interval = calculate_confidence_interval(list_avg_accuracy)
        print(f"Hyperparameters: K={k}, Distance={similarity_function.__name__}")
        print(f"Confidence Interval: {confidence_interval[1]:.4f}, {confidence_interval[2]:.4f}")
        print("=" * 100)

        #Updating the best hyperparameters if the current set performs better
        if confidence_interval[0] > best_accuracy:
            best_accuracy = confidence_interval[0]
            best_hyperparameters = {'K': k, 'Distance': similarity_function.__name__}

# Printing the best hyperparameters and corresponding accuracy
print(f"Best Hyperparameter: K={best_hyperparameters['K']}, Distance={best_hyperparameters['Distance']}, Accuracy: {best_accuracy:.4f}")
