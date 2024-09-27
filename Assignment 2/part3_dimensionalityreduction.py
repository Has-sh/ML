import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

# Load the dataset from a pickle file
dataset = pickle.load(open("data/part3_dataset.data", "rb"))


def plot_results(dataset, title):
    # Create a 1x2 subplot for the 2D scatter plots
    _, (axis1, _) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create a scatter plot for the 2D representation of the dataset
    
    scatter = axis1.scatter(dataset[:, 0], dataset[:, 1])
    #plt.colorbar(scatter, ax=axis1)
    axis1.set_title(f'{title} - 2D Scatter Plot')

    plt.show()

def run(dataset, method):
    # Choose the dimensionality reduction method (t-SNE or UMAP)
    if method == 'tsne':
        plot = TSNE(n_components=2)
    elif method == 'umap':
        plot = UMAP(n_components=2)
    
    # Perform dimensionality reduction
    ploted = plot.fit_transform(dataset)
    return ploted

# Run t-SNE on the dataset and plot the results
tsne_dataset1 = run(dataset, method='tsne')
plot_results(tsne_dataset1, 'Dataset t-SNE')

# Run UMAP on the dataset and plot the results
umap_dataset1 = run(dataset, method='umap')
plot_results(umap_dataset1, 'Dataset UMAP')
