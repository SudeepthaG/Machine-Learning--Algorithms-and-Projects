

from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    # Reading input data
    input_data = pd.read_csv("linsep.txt", sep=',', header=None)
    dataarray = np.array(input_data)


    # assigning points and labels
    Y= np.array(dataarray)[:, 2]
    X = np.array(dataarray)[:, 0:2]
    # print(X)
    # print(Y)

    # finding margin line
    linsep_svm=svm.SVC(gamma='auto',kernel="linear")
    linsep_svm.fit(X,Y)
    print("Intercept:")
    print(linsep_svm.intercept_)
    print("Weights:")
    print(linsep_svm.coef_[0])
    print("Support vectors:")
    print(linsep_svm.support_vectors_)

    # plotting data and margin line
    plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',alpha=1,s=50,edgecolors='k')
    x2_lefttarget = -(linsep_svm.coef_[0][0]*(-1)+linsep_svm.intercept_)/linsep_svm.coef_[0][1]
    x2_righttarget = -(linsep_svm.coef_[0][0]*(1)+linsep_svm.intercept_)/linsep_svm.coef_[0][1]
    plt.scatter(linsep_svm.support_vectors_[:,0],linsep_svm.support_vectors_[:,1],facecolors='none',s=100, edgecolors='k')
    plt.plot([-1,1], [x2_lefttarget,x2_righttarget])
    plt.show()

if __name__ == "__main__":
    main()
