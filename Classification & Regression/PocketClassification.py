
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Pocket:
    def __init__(self, data, labels):
        self.X = data
        self.Y = labels
        self.dataSize = np.size(data,0)
        self.weights = np.random.rand(np.size(data,1))
        self.pocketWeight = []
        self.misclassified = []
        self.iterationNum= []
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
        iterations = 0
        self.misclassified.append(float('inf'))
        self.iterationNum.append(iterations)
        while violated > 0 and iterations < 7000:
            violated = 0
            iterations += 1
            for i in range(self.dataSize):
                wX = np.dot(self.X[i], self.weights)
                if wX < 0 and self.Y[i] == 1:
                    self.weights += self.alpha * self.X[i]
                    violated += 1
                elif wX >= 0 and self.Y[i] == -1:
                    self.weights -= self.alpha * self.X[i]
                    violated += 1
            prediction = self.predict()
            misclassifications = self.dataSize - prediction
            if misclassifications < self.misclassified[-1]:
                self.pocketWeight = self.weights
            self.misclassified.append(misclassifications)
            self.iterationNum.append(iterations)
            
        return self.pocketWeight
    
    def plot(self):
        plt.plot(self.iterationNum, self.misclassified)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Misclassifications")
        plt.show()

def parseInput():
    input = pd.read_csv('classification.txt', usecols=[0,1,2,4], names=['x','y','z','label'])
    labels = input['label'].to_numpy()
    data = input[['x','y','z']]
    # add the bias to training data
    data.insert(0, 'bias', 1)
    data = data.to_numpy()
    return data, labels

if __name__ == '__main__':
    data, labels = parseInput()
    p = Pocket(data, labels)
    weights = p.train()
    total = np.size(data,0)
    correct = p.predict()
    print("Weights:\n", weights)
    print("\nAccuracy:\n", (correct/total)*100,"%")
    p.plot()
