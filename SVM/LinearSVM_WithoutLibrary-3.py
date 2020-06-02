# Group Members:
# Vanessa Tan (ID: 4233243951)
# Sudeeptha Mouni Ganji (ID: 2942771049)

import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

class SVM:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.size = len(X)

    def linear(self):
        q = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                q[i,j] = np.dot(self.X[i], self.X[j])
        # construct matrices for qpp
        Q = matrix(np.outer(self.Y,self.Y) * q)
        p = matrix(np.full(self.size, -1.0))
        G = matrix(np.diag(np.full(self.size, -1.0)))
        h = matrix(np.zeros(self.size))
        A = matrix(self.Y*1.0, (1,self.size))
        b = matrix(0.0)  
        # find the alphas
        solution = np.ravel(solvers.qp(Q, p, G, h, A, b)['x'])
        # calculate the weight vector
        weights = sum(solution.reshape(self.size,1) * self.Y.reshape(self.size,1) * self.X)
        
        # use alpha > 0.00001 to find intercept and support vectors
        indices = np.where(solution > 1e-5)[0]
        intercept = 1.0/self.Y[indices[0]] - np.dot(weights, self.X[indices[0]])
        print("\nWeights:", weights)
        print("Intercept:", intercept) 
        eq = ""
        for i in range(weights.size):
            eq += str(round(weights[i], 5)) + "x"+ str(i+1) + " "
        eq += str(round(intercept,5)) + " = 0"
        print("\nEquation Line:")
        print(eq)
        
        supportVectors = self.X[indices]
        print("\nSupport Vectors:")
        print(supportVectors)



def parseInput():
    # separate input data and labels
    df = pd.read_csv('linsep.txt', names=['X', 'Y', 'labels'])
    data = df[['X','Y']].to_numpy()
    labels = df['labels'].to_numpy()
    return data, labels

def main():
    X, Y = parseInput()
    svm = SVM(X,Y)
    svm.linear()

if __name__ == "__main__":
	main()