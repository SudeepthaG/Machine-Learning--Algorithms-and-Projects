

import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Reading and assigning input data as per given instructions
    inputdata = pd.read_csv("classification.txt", delimiter=',')
    dataarray = np.array(inputdata)
    Y_train = np.array(dataarray)[:,4]
    X = np.array(dataarray)[:, 0:3]

    # Adding new column X0=1 at beginning
    n, m = X.shape
    X0 = np.ones((n, 1))
    X_i = np.hstack((X0, X))

    #Assigning values
    iterations=7000
    gradient=np.zeros(4)
    weights=np.random.rand(X_i.shape[1])
    learning_rate=0.01

    # Computing weights
    for value in range(iterations):
        for i in range(len(Y_train)):
            denominator= 1+np.exp(np.dot(np.dot(X_i[i],weights.T),Y_train[i]))
            gradient= gradient + (np.dot(Y_train[i],X_i[i])/denominator)
        gradient=-(gradient/len(Y_train))
        weights=weights-(learning_rate*gradient)
        S=np.dot(X_i,weights.T)

    # Assigning probabilities and labels
    probability=(np.exp(S))/(1+np.exp(S))
    probability[probability > 0.5] = 1
    probability[probability < 0.5] = -1

    # Computing accuracy
    count = 0.0
    for i in range (len(Y_train)):
        if probability[i]==Y_train[i]:
            count+=1
    accuracy=(count/len(Y_train))

    print("Accuracy:",accuracy)
    print("Weights:",weights)
