

from collections import Counter
import math
import numpy as np
from sklearn.cluster import KMeans


class EMalgorithmImplementation:

    def __init__(self, no_of_clusters, points):
        self.no_of_clusters = no_of_clusters
        self.points = points

        # using kmeans for getting initial points to improve accuracy and efficieny
        kmeans = KMeans(n_clusters=self.no_of_clusters)
        kmeans.fit(self.points)

        # initializing AMPLITUDE(alpha 𝝰)
        self.amplitude = []
        labels = kmeans.labels_
        for i in range(self.no_of_clusters):
            self.amplitude.append((Counter(labels).get(i)) / 150)
        self.amplitude = np.array(self.amplitude)

        # initializing MEANS(mu µ )
        self.means = kmeans.cluster_centers_
        self.means = np.array(kmeans.cluster_centers_)

        # initializing COVARIANCE MATRIX(sigma 𝜮)
        self.covariance = np.array(
            np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]))

        # initializing THRESHOLD AND WEIGHTS OF x_i
        self.likelihood_threshold = 1e-3
        self.likelihood = None
        self.wts = np.zeros([self.no_of_clusters, 150])

        self.implementEM()



    # IMPLEMENTING ITERATIVE FUNCTION FOR CALLING E_step & M_step TILL CONVERGENCE OF LIKELIHOOD
    def implementEM(self):
        iterations = 0
        new_likelihood = self.calculate_new_likelihood()
        while ((iterations == 0) or (new_likelihood - self.likelihood > self.likelihood_threshold)):
            self.likelihood = new_likelihood
            self.E_step()
            self.M_step()
            new_likelihood = self.calculate_new_likelihood()
            iterations += 1



    # CALCULATING new likelihood AFTER PERFORMING E_step & M_step
    def calculate_new_likelihood(self):
        new_likelihood = 0
        for i in range(150):
            temp = 0
            for j in range(self.no_of_clusters):
                x_i = 1 / pow((2 * math.pi), -(self.points.shape[1]) / 2) * pow(abs(np.linalg.det(self.covariance[j])),-1 / 2) * np.exp(
                    -1 / 2 * np.dot(np.dot((self.points[i].T - self.means[j].T).T, np.linalg.inv(self.covariance[j])),
                                    (self.points[i].T - self.means[j].T)))
                temp += self.amplitude[j] * x_i
            new_likelihood += np.log(temp)
        return new_likelihood



    # IMPELEMENTING EXPECTATION STEP
    def E_step(self):
        s = np.zeros(150)
        for i in range(150):
            temp = np.zeros(self.no_of_clusters)
            for j in range(self.no_of_clusters):
                x_i = 1 / pow((2 * math.pi), -(self.points.shape[1]) / 2) * pow(
                    abs(np.linalg.det(self.covariance[j])), -1 / 2) * np.exp(-1 / 2 * np.dot(
                    np.dot((self.points[i].T - self.means[j].T).T, np.linalg.inv(self.covariance[j])),
                    (self.points[i].T - self.means[j].T)))
                temp[j] = float(self.amplitude[j]) * x_i
                s[i] += temp[j]
            for k in range(self.no_of_clusters):
                self.wts[k][i] = temp[k] / s[i]



    # IMPLEMENTING MAXIMIZATION STEP
    def M_step(self):
        for k in range(self.no_of_clusters):
            # Calculating amplitude[k]
            self.amplitude[k] = np.sum(self.wts[k]) / 150

            # Calculating means[k]
            total = np.zeros(self.means.shape[1])
            for i in range(150):
                total += self.wts[k][i] * self.points[i]
            self.means[k] = total / np.sum(self.wts[k])

            # Calculating covariance[k]
            summation = np.zeros([self.points.shape[1], self.points.shape[1]])
            for i in range(150):
                if self.points[i].ndim == 1:

                    #converting points and means which are in 1-d array to 2*1 matrix for calculation purposes
                    points_temp = self.points[i].reshape(self.points.shape[1], 1)
                    means_temp = self.means[k].reshape(self.means.shape[1], 1)
                    diff_temp = points_temp - means_temp
                    summation += self.wts[k][i] * np.dot(diff_temp, diff_temp.T)
                else:
                    summation += self.wts[k][i] * np.dot(self.means[i] - self.means[i], (self.points[i] - self.means[i]).T)
            self.covariance[k] = summation / np.sum(self.wts[k])



if __name__ == '__main__':
    df = np.loadtxt("clusters.txt", delimiter=',')
    em = EMalgorithmImplementation(3, df)
    print("The means are:\n", em.means)
    print("\nThe amplitudes are:\n", em.amplitude)
    print("\nThe covariance matrix is:\n", em.covariance)
