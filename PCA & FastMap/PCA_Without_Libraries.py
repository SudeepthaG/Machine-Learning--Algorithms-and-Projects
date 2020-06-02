
import numpy as np
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, datapoints=None, dimensions=0):
        self.x = np.array(datapoints)
        self.k = dimensions
        self.covar = np.array
        self.mn_x = np.array
        self.eigenvalue = np.array
        self.eigenvector = np.array
        self.sorted_eigenvalue = np.array
        self.sorted_eigenvector = np.array
        self.sorted_k_eigenvector = np.array
        self.reduced_2D_data_T = np.array


    # Calcuating mean and Normalizing data on mean
    def find_mean_normalization(self, x):
        _mean = np.mean(x, axis=0)
        self.mn_x = np.array(x - _mean)
        return self.mn_x


    # Calculating covariance matrix on mean normalized data
    def calculate_covariance_matrix(self, x):
        _x = np.array(x)
        covar = np.cov(_x.T)
        self.covar = covar
        return self.covar


    # Sorting the eigen vectors so as to get first 2 principal components
    def get_sorted_eigenvector_nk(self, covariance, k):
        _v = np.array
        eigenvector, eigenvalue, _v = np.linalg.svd(covariance)
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector
        sorted_value_idx = np.argsort(-eigenvalue)
        eigenvector = eigenvector[:, sorted_value_idx]
        self.sorted_eigenvector = eigenvector
        eigenvalue = eigenvalue[sorted_value_idx]
        self.sorted_eigenvalue = eigenvalue
        self.sorted_k_eigenvector = self.sorted_eigenvector[:, :k]
        return self.sorted_k_eigenvector


    # Converting data format for calculation purposes
    def k_dimension_projection(self, v, x):
        reduced_2D_data = v.T.dot(x.T)
        return reduced_2D_data



    # IMPLEMENTING PCA ALGORITHM
    def implementPCA(self, data, dimensions):
        self.x = data
        self.k = dimensions
        if (data is None):
            return None
        else:
            nx = self.find_mean_normalization(self.x)
            covar = self.calculate_covariance_matrix(nx)
            v_k = self.get_sorted_eigenvector_nk(covar, self.k)
            reduced_2D_data = self.k_dimension_projection(v_k, nx)
            self.reduced_2D_data_T = reduced_2D_data.T
            return self.reduced_2D_data_T



if __name__ == '__main__':

    dimensions = 2
    # Reading input data
    data = np.genfromtxt('pca-data.txt',delimiter='\t')

    # Running pca algorithm
    pca = PCA()
    reduced_data = pca.implementPCA(data, dimensions)
    print('The new 2-dimensional data after reduction is:\n', reduced_data)
    print('Directions of first two principal components(Eigenvectors:)\n', pca.sorted_k_eigenvector.T.transpose())

    # Plotting eigenvectors to graph
    plt.plot(pca.sorted_k_eigenvector.T.transpose())
    plt.show()

