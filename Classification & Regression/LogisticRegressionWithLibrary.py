# Group Members:
# Vanessa Tan (ID: 4233243951)
# Sudeeptha Mouni Ganji (ID: 2942771049)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':

    # Reading and assigning input data
    input_data = pd.read_csv("classification.txt", delimiter=',')
    input_data=np.array(input_data)
    X = input_data[:, :3]
    Y = input_data[:, 4]
    X = np.c_[np.ones(len(X)), np.array(X)]

    # Fitting data to logistic regression algorithm
    lr = LogisticRegression()
    lr = lr.fit(X, Y)

    # Calculating and printing required output
    score = lr.score(X, Y)
    W = lr.coef_
    print('Accuracy:', score)
    print ('Weights:', W )


