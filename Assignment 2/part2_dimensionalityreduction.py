import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

def plot_results(dataset, title):
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

# Load datasets from pickle files
dataset1 = pickle.load(open("data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("data/part2_dataset_2.data", "rb"))

# Run t-SNE and UMAP on Dataset 1
tsne_dataset1 = run(dataset1, method='tsne')
plot_results(tsne_dataset1, 'Dataset 1 t-SNE')

umap_dataset1 = run(dataset1, method='umap')
plot_results(umap_dataset1, 'Dataset 1 UMAP')

# Run t-SNE and UMAP on Dataset 2
tsne_dataset2 = run(dataset2, method='tsne')
plot_results(tsne_dataset2, 'Dataset 2 t-SNE')

umap_dataset2 = run(dataset2, method='umap')
plot_results(umap_dataset2, 'Dataset 2 UMAP')
