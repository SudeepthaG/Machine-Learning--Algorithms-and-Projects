# Group Members:
# Vanessa Tan (ID: 4233243951)
# Sudeeptha Mouni Ganji (ID: 2942771049)

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron

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
    p = Perceptron(max_iter=7000)
    p.fit(data, labels)
    prediction = p.predict(data)

    correct = np.sum(prediction == labels)
    total = np.size(data,0)

    print("Weights:\n", p.coef_[0])
    print("\nAccuracy:\n", (correct/total)*100,"%")