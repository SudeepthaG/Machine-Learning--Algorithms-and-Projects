

import pandas as pd
from sklearn.cluster import KMeans

def kmeans():
    df = pd.read_csv('clusters.txt', names=['x','y'])
    # initialize the KMeans class with the # of clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_
    print("Centroids:")
    for i in centroids:
        print("({}, {})".format(i[0], i[1]))


if __name__ == '__main__':
    kmeans()
