


import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Reading and manipulating input data
    input_data = pd.read_csv("linear-regression.txt", sep=',', header=None)
    dataarray = np.array(input_data)

    Y_vector = np.array(dataarray)[:, 2]
    X = np.array(dataarray)[:, 0:2]
    n, m = X.shape
    X0 = np.ones((n, 1))
    D = np.hstack((X0, X))


    # Calculating weights and weighted_x
    weights = np.dot(np.dot(np.linalg.inv(np.dot(D.T, D)), D.T), Y_vector)
    weighted_X = np.dot(D, weights)

    # Printing weighted x
    print("weighted X:")
    print(weighted_X)

    # Printing weights
    print("Weights after final iteration: \nIntercept and Coefficients")
    print(weights)

