import pandas as pd
import random
import copy
import math
import matplotlib.pyplot as plt

class FastMap:
    def __init__(self, dist, words):
        self.dist = dist
        self.words = words
        self.numOfObjects = len(words)
        self.results = {i:[] for i in range(self.numOfObjects)}

    def findFurthestObject(self):
        # use pivot changing heuristic
        pivot = random.randint(0,self.numOfObjects-1)

        while True:
            # find the object with the max dist from the pivot
            maxDist = max(self.dist[pivot])
            idxOfMax = self.dist[pivot].index(maxDist)

            maxDist2 = max(self.dist[idxOfMax])
            idxOfMax2 = self.dist[idxOfMax].index(maxDist2)

            # keep looping until the points converge 
            if idxOfMax2 == pivot and maxDist == maxDist2:
                # pivots converged
                break
            else:
                pivot = idxOfMax
        return min(idxOfMax,pivot),max(idxOfMax,pivot)
    
    def updateDist(self):
        # update the distance matrix with the new calculated distances
        newDist = [[0 for i in range(self.numOfObjects)] for j in range(self.numOfObjects)]
        for i in range(self.numOfObjects):
            for j in range(self.numOfObjects):
                newDist[i][j] = math.sqrt((self.dist[i][j])**2 - (self.results[i][-1]-self.results[j][-1])**2)
        self.dist = newDist
    
    def FastMapAlgorithm(self, k):
        if k <= 0:
            return
        a,b = self.findFurthestObject()
        for i in range(self.numOfObjects):
            if i == a:
                new_dist = 0
            elif i == b:
                new_dist = self.dist[a][b]
            else:
                # find the triangular distance between object a,b and i
                new_dist = (self.dist[a][i]**2 + self.dist[a][b]**2 - self.dist[b][i]**2)/(2*self.dist[a][b])
            
            self.results[i].append(new_dist)
        self.updateDist()
        self.FastMapAlgorithm(k-1)

    def plotResults(self):
        for i,pt in self.results.items():
            print(self.words[i],"-->", pt)
            plt.scatter(pt[0], pt[1])
            plt.annotate(self.words[i], (pt))
        plt.show()
        return

def parseInput():
    points = pd.read_csv('fastmap-data.txt', sep='\t', names=['x','y','dist'])
    wordlist = pd.read_csv('fastmap-wordlist.txt', names=['words'])
    wordlist = wordlist.words.tolist()
    numOfObjects = len(wordlist)
    dist = [[0 for i in range(numOfObjects)] for j in range(numOfObjects)]
    for i, row in points.iterrows():
        dist[row['x']-1][row['y']-1] = row['dist']
        dist[row['y']-1][row['x']-1] = row['dist']
    return dist, wordlist

if __name__ == '__main__':
    dist, words = parseInput()
    fastmap = FastMap(dist, words)
    fastmap.FastMapAlgorithm(2)
    fastmap.plotResults()