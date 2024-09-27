import numpy as np
import pickle
import matplotlib.pyplot as plt
from Kmeans import KMeans
import time

def confidence_interval(min_loss):
    # Calculate the mean and margin of error for a list of minimum losses
    mean_val = np.mean(min_loss)
    margin_of_error = np.std(min_loss) * 1.96 / np.sqrt(len(min_loss))
    return mean_val, mean_val - margin_of_error, mean_val + margin_of_error

def process_dataset(dataset, dataset_num):
    # Define a range of K values    
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    loss = []
    confidence_intervals = []
    for k in K:
        min_loss=[]
        running_times = []
        for _ in range(10):
            current_losses = []
            current_running_times = []
            for _ in range(10):
                # Create a new instance of the KMeans class for each iteration
                kmeans_instance = KMeans(dataset, k)
                start_time = time.time()
                _, _, current_loss = kmeans_instance.run()
                end_time = time.time()
                current_losses.append(current_loss)
                current_running_times.append(end_time - start_time)

            min_loss.append(min(current_losses))
            running_times.append(np.mean(current_running_times))

        # Calculate average minimum loss and confidence intervals
        avg_min_loss, lower_bound, upper_bound = confidence_interval(min_loss)
        loss.append(np.mean(avg_min_loss))
        confidence_intervals.append((lower_bound, upper_bound))
        avg_running_time = np.mean(running_times)

        # Print and display results
        print(f"Dataset: {dataset_num} - K: {k}")
        print(f"Avg of Loss: {np.mean(min_loss)}")
        print(f"Confidence Interval: ({lower_bound}, {upper_bound})")
        print(f"Avg Running Time: {avg_running_time} seconds")
        print("------------------------------------------------------------")

    # Plot the results    
    plt.errorbar(K, loss, yerr=np.transpose(np.array(confidence_intervals)), fmt='o-', capsize=5, label='Confidence Interval')
    plt.plot(K, loss)
    plt.title(f'Loss vs K for Dataset {dataset_num}')
    plt.xlabel('Number of Clusters(K)')
    plt.ylabel('Average Loss')
    plt.show()

# Load datasets from pickle files
dataset1 = pickle.load(open("data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("data/part2_dataset_2.data", "rb"))

# Process and visualize results for each dataset
process_dataset(dataset1, 1)
process_dataset(dataset2, 2)
