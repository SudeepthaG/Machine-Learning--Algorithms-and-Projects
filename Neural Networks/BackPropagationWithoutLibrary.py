

import re
import numpy as np
from skimage import io
import math

# Reading training data
def read_training_data():
    files_one = []
    files_zero = []
    with open('downgesture_train.list') as f:
        for line in f.readlines():
            if (re.search(r'down', line)):
                files_one.append(line.strip())
            else:
                files_zero.append(line.strip())
    return files_one, files_zero

# Reading test data
def read_test_data():
    files_one = []
    files_zero = []
    with open('downgesture_test.list') as f:
        for line in f.readlines():
            if (re.search(r'down', line)):
                files_one.append(line.strip())
            else:
                files_zero.append(line.strip())
    return files_one, files_zero


# Reading gestures
def parse_picture(file_lists, label):
    data = []
    for filename in file_lists:
        img =io.imread(filename, -1)
        reshaped_img = list(img.reshape(len(img) * len(img[0])))
        reshaped_img.append(label)
        data.append(reshaped_img)
    return data


# iSplitting input matrix into two X,Y matrices
def split_data(data):
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


class NN(object):
    W = []
    X = []
    S = []
    Theta = []
    learning_rate = 0.1

    def __init__(self, layers, X, Y):
        self.layers = layers
        self.n = len(layers)
        self.data_x_n, self.data_x_m = np.shape(X)
        for i in range(self.n):
            if (i == 0):
                self.X.append(X)
                self.Y = Y[:, np.newaxis]
                self.S.append([])
                self.Theta.append([])
            else:
                self.X.append(np.random.randn(self.data_x_n, layers[i]))
                self.S.append(np.random.randn(self.data_x_n, layers[i]))
                self.Theta.append(np.random.randn(self.data_x_n, layers[i]))
            if (i != self.n - 1):
                self.W.append(self.random_wts(layers[i], layers[i + 1]))


    def random_wts(self, m, n):
        return np.random.rand(m, n) * 0.02 - 0.01

    def set_data(self, X, Y):
        self.X[0] = X
        self.Y = Y[:, np.newaxis]

    def predict(self):
        for i in range(1, self.n):
            self.S[i] = np.dot(self.X[i - 1], self.W[i - 1])
            np.clip(self.S[i], -700, None, out=self.S[i])
            self.X[i] = 1/(1+np.exp(-self.S[i]))   #sigmoid function  Ɵ(s) = 1/(1+e^-s)


    def backpropapagation(self):
        for i in range(self.n - 1, 0, -1):
            if (i == self.n - 1):
                self.Theta[i] = np.multiply(np.multiply(self.X[i] - self.Y, 2), np.multiply(self.X[i], (1 - self.X[i])))
            else:

                self.Theta[i] = np.multiply(np.dot(self.Theta[i + 1], self.W[i].T),
                                            np.multiply(self.X[i], (1 - self.X[i])))
        for i in range(self.n - 1):
            self.W[i] = self.W[i] - np.multiply(self.learning_rate,
                                                np.dot(self.X[i].T, self.Theta[i + 1]) / self.data_x_n)


def main():
    # building neural network based on training data
    files_one, files_zero = read_training_data()
    data_one = parse_picture(files_one, 1)
    data_zero = parse_picture(files_zero, 0)
    data = np.concatenate((data_one, data_zero), axis=0)
    np.random.shuffle(data)
    X, Y = split_data(data)
    nn = NN([960, 100, 1], X, Y)
    for i in range(1000):  #as given epoch=1000
        nn.predict()
        nn.backpropapagation()
    predict = np.ones((len(data), 1))
    predict[nn.X[2] < 0.5] = 0
    acc = 0
    for i in range(len(data)):
        if (int(predict[i, 0]) == Y[i]):
            acc = acc + 1
    print("Accuracy on training data:")
    print(acc * 1.0 / len(data))

    # performing predictions on test data based on above neural network
    files_one_test, files_zero_test = read_test_data()
    data_one_test = parse_picture(files_one_test, 1)
    data_zero_test = parse_picture(files_zero_test, 0)
    data_test = np.concatenate((data_one_test, data_zero_test), axis=0)
    X_test, Y_test = split_data(data_test)
    nn.set_data(X_test, Y_test)
    nn.predict()
    predict_test = np.ones((len(data_test), 1))
    predict_test[nn.X[2] < 0.5] = 0

    #printing final outputs
    print("Prediction on test data:")
    print(predict_test[:, 0])
    acc_test = 0
    for i in range(len(data_test)):
        if (int(predict_test[i, 0]) == Y[i]):
            acc_test = acc_test + 1
    print("Accuracy on test data:")
    print(acc_test * 1.0 / len(data_test))


if __name__=="__main__":
    main()
