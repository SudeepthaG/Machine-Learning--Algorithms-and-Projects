

import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

class SVM:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.size = len(X)

    def nonlinear(self):
        q = np.zeros((self.size, self.size))
        kernelX = np.zeros((self.size, 6))
        for i in range(self.size):
            # nonlinear transformation for polynomial kernel 
            kernelX[i] = ([1.0, self.X[i][0]**2, self.X[i][1]**2, self.X[i][0]*np.sqrt(2),  self.X[i][1]*np.sqrt(2), self.X[i][0]*self.X[i][1]*np.sqrt(2)])
        for i in range(self.size):
            for j in range(self.size):
                q[i,j] = np.dot(kernelX[i], kernelX[j])
        # construct matrices for qpp
        Q = matrix(np.outer(self.Y,self.Y) * q)
        p = matrix(np.full(self.size, -1.0))
        G = matrix(np.diag(np.full(self.size, -1.0)))
        h = matrix(np.zeros(self.size))
        A = matrix(self.Y*1.0, (1,self.size))
        b = matrix(0.0)  
        # find the alphas
        solution = np.ravel(solvers.qp(Q, p, G, h, A, b)['x'])
        
        # use alpha > 0.00001    
        indices = np.where(solution > 1e-5)[0]
        alphas = solution[indices]
        Xn = self.X[indices]
        Yn = self.Y[indices]
        Zn = kernelX[indices]

        weights = sum(alphas.reshape(alphas.size,1) * Yn.reshape(Yn.size,1) * Zn)
        print("\nWeights:")
        print(weights)
        b = 0
        for i in range(alphas.size):
            b += Yn[i]
            b -= sum(alphas * Yn * q[indices[i], indices])
        b /= alphas.size

        print("Intercept:", b)
        print("\nKernel function: Polynomial Kernel")
        print("\nSupport Vectors:")
        print(Xn)
        

def parseInput():
    df = pd.read_csv('nonlinsep.txt', names=['X', 'Y', 'labels'])
    data = df[['X','Y']].to_numpy()
    labels = df['labels'].to_numpy()
    return data, labels

def main():
    X, Y = parseInput()
    svm = SVM(X,Y)
    svm.nonlinear()

if __name__ == "__main__":
	main()
