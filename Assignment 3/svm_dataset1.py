import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC


dataset, labels = pickle.load(open("HW 3 Data and Source Codes/data/part2_dataset1.data", "rb"))

configs=[
    {'C': 1, 'kernel': 'poly', 'degree': 2},
    {'C': 1, 'kernel': 'rbf', 'gamma': 'auto'},
    {'C': 10, 'kernel': 'poly', 'degree': 2},
    {'C': 10, 'kernel': 'rbf', 'gamma': 'auto'}
]

plt.figure(figsize=(12, 8))

for i, config in enumerate(configs, 1):
    svm = SVC(**config)
    svm.fit(dataset, labels)
    plt.subplot(2, 2, i)
    h = .01  # Step size in the mesh
    x_min, x_max = dataset[:, 0].min()-0.5, dataset[:, 0].max()+0.5
    y_min, y_max = dataset[:, 1].min()-0.5, dataset[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])#used for boundries
    # Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.scatter(dataset[:,0], dataset[:,1], cmap="Paired_r", c=labels, edgecolors='k')
    plt.title(f"C={config['C']}, Kernel={config['kernel']}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
plt.tight_layout()
plt.show()
