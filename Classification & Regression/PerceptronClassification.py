

import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, data, labels):
        self.X = data
        self.Y = labels
        self.dataSize = np.size(data,0)
        self.weights = np.random.rand(np.size(data,1))
        self.alpha = 0.0001

    def classify(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def predict(self):
        # predict output by getting the dot product of data and weights
        classifyOutput = np.vectorize(self.classify)
        prediction = classifyOutput(np.dot(self.X, self.weights))
        correct = np.sum(self.Y == prediction)
        return correct

    def train(self):
        violated = self.dataSize
        # update weights until there are no violated constraints
        while violated > 0:
            violated = 0
            for i in range(self.dataSize):
                wX = np.dot(self.X[i], self.weights)
                if wX < 0 and self.Y[i] == 1:
                    self.weights += self.alpha * self.X[i]
                    violated += 1
                elif wX >= 0 and self.Y[i] == -1:
                    self.weights -= self.alpha * self.X[i]
                    violated += 1
        return self.weights


def parseInput():
    input = pd.read_csv('classification.txt', usecols=[0,1,2,3], names=['x','y','z','label'])
    labels = input['label'].to_numpy()
    data = input[['x','y','z']]
    # add the bias to training data
    data.insert(0, 'bias', 1)
    data = data.to_numpy()
    return data, labels

if __name__ == '__main__':
    data, labels = parseInput()
    total = np.size(data,0)
    p = Perceptron(data, labels)
    weights = p.train()
    correct = p.predict()
    print("Weights:\n", weights)
    print("\nAccuracy:\n", (correct/total)*100,"%")
