
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Plotting the data in 3-D
def plot(X, Z, lr):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    x = X[:, 0]
    y = X[:, 1]
    z = np.array(Z[:])

    # Plotting points
    ax.scatter(x, y, z, c='g', marker='o')
    xx, yy = np.meshgrid(np.arange(x.min() - 0.2, x.max() + 0.2, 0.02), np.arange(y.min() - 0.2, y.max() + 0.2, 0.02))
    zz = np.zeros(shape=(xx.shape))
    for i in range(len(xx)):
        for j in range(len(xx[i])):
            zz[i][j] = lr.predict([[xx[i][j], yy[i][j]]])

    # Plotting the segmentation plane
    ax.plot_surface(xx, yy, zz, color='y', alpha=0.3)
    plt.show()


if __name__ == '__main__':

    # reading and assigning variables
    input_data=np.genfromtxt("linear-regression.txt",delimiter=',')
    X=input_data[:, :2]
    Z=input_data[:, 2]

    # fitting the given input to linearRegression function
    lr = LinearRegression()
    lr.fit(X, Z)

    # printing output
    print("Weights after final iteration:")
    print("Coefficents are: ")
    print(lr.coef_)
    print("\n\nIntercept is: ")
    print(lr.intercept_)

    # Plotting data
    plot(X, Z, lr)
