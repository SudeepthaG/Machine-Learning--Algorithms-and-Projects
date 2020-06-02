

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class EMalgorithm:

    def __init__(self, k, points):
        self.k = k
        self.points = points


    def implementEM(self):
        k=self.k
        points=self.points
        plt.scatter(points[0], points[1])
        gmm = GaussianMixture(n_components=k)
        gmm.fit(points)
        labels = gmm.predict(points)
        points['labels'] = labels

        #Giving labels to each sample cluster
        d0 = points[points['labels'] == 0]
        d1 = points[points['labels'] == 1]
        d2 = points[points['labels'] == 2]

        #Plotting clusters to same plot
        plt.scatter(d0[0], d0[1], c='r')
        plt.scatter(d1[0], d1[1], c='yellow')
        plt.scatter(d2[0], d2[1], c='g')

        #Plotting means of clusters to same plot
        means=gmm.means_
        plt.scatter(means[:, 0], means[:, 1],c='blue', s=50)

        plt.show()

        #Printing the required values
        print("The means are:")
        print(gmm.means_)

        print("\nThe amplitudes are:")
        print(gmm.weights_)

        print("\nThe covariance matrix is:")
        print(gmm.covariances_)


def parseInput():
    df = pd.read_csv('clusters.txt', names=[0,1])
    return df


if __name__ == '__main__':
    df = parseInput()
    em = EMalgorithm(3, df)
    em.implementEM()
