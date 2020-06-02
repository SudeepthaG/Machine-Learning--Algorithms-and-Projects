

from sklearn import svm
import numpy as np
import pandas as pd

def main():

    # Reading input data
    input_data = pd.read_csv("nonlinsep.txt", sep=',', header=None)
    dataarray = np.array(input_data)

    # assigning points and labels
    Y= np.array(dataarray)[:, 2]
    X = np.array(dataarray)[:, 0:2]
    # print(X)
    # print(Y)

    # fitting data to svm
    non_linsep_svm = svm.SVC(kernel='poly', degree=2)
    non_linsep_svm.fit(X, Y)

    # printing coordinates 
    print("Intercept:")
    print(non_linsep_svm.intercept_)
    print("Weights:")
    print(non_linsep_svm.dual_coef_[0])
    print("Support vectors:")
    print(non_linsep_svm.support_vectors_)


if __name__ == "__main__":
    main()
