# Group Members:
# Vanessa Tan (ID: 4233243951)
# Sudeeptha Mouni Ganji (ID: 2942771049)

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# loading and converting input data
data = np.loadtxt("pca-data.txt", delimiter = "\t")
data = np.mat(data)
mean = np.mean(data, axis = 0)
data = data - mean

# implementing pca algorithm
pca=PCA(n_components=2)
pca.fit(data)
print("Direction of first two principal components(Eigenvectors) are:")
print(pca.components_.T)


# plotting output to graph
plt.plot(pca.components_.T)
plt.show()
