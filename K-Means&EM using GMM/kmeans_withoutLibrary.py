

import pandas as pd
import random
import math
import statistics

class kMeans:
    def __init__(self, k, points):
        self.k = k
        self.points = points

    def initialize(self):
        # find the min and max of data points
        minVal = min(self.points.min())
        maxVal = max(self.points.max())
        centroids = {}

        # assign random centroids within the range of the min and max
        for i in range(self.k):
            centroids[i] = [random.uniform(minVal, maxVal), random.uniform(minVal, maxVal)]
        return centroids

    def findClosestCentroid(self, points, centroids):
        # compute the distance of every point to every cluster centroid
        for i, pt in centroids.items():
            points[i] = ((points['x']-pt[0])**2 + (points['y']-pt[1])**2).apply(math.sqrt)

        # find the closest centroid and assign it to that point
        points['closest_centroid'] = points[[0,1,2]].idxmin(axis=1)
        return points

    def updateCentroids(self, points, centroids):
        new_centroids = centroids.copy()
        for i in centroids.keys():
            # get all the points in cluster i
            new = points.loc[points['closest_centroid'] == i]
            if len(new) > 0:
                # update the new centroid to be the mean of cluster
                new_x = statistics.mean(new['x'])
                new_y = statistics.mean(new['y'])
                new_centroids[i] = [new_x, new_y]
        return new_centroids

    def kmeansAlgorithm(self):
        centroids = self.initialize()
        # continue looping until convergence
        while True:
            old_centroids = centroids
            data = self.findClosestCentroid(self.points, centroids)
            centroids = self.updateCentroids(data, centroids)
            if old_centroids == centroids:
                print("Centroids")
                for i in centroids.values():
                    print("({}, {})".format(i[0], i[1]))
                break

def parseInput():
    df = pd.read_csv('clusters.txt', names=['x','y'])
    return df


if __name__ == '__main__':
    df = parseInput()
    kmeans = kMeans(3, df)
    kmeans.kmeansAlgorithm()
