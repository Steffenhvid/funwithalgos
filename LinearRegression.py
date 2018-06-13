from SupervisedLearning import SupervisedLearning
import numpy as np

class LinearRegression(SupervisedLearning):

    def MSE(self, X, y, theta):
        '''
        Cost function (Mean Squarred Error)
        :param y_hat: np.array (n,) - The predictions w.X
        :param y: np.array (n,) - The target
        :return: The average error - the smaller the better
        '''
        m = len(y)
        J = np.sum((X.dot(theta) - y) ** 2) / 2 / m
        return J


    def gradient_descent(self, X, y, theta=None, lr=0.1, iterations=1000, epsilon=0.0000000001):
        '''

        :param X: np.array (n,d)
        :param w: np.array (d,)
        :param lr: scalar, learning rate
        :return:
        '''
        n = len(y)
        if theta is None:
            theta = np.zeros(X.shape[1])
        cost = np.zeros(iterations)
        for i in range(iterations):
            theta -= (lr/n) * sum((X.dot(theta) - y).dot(X))
            cost[i] = self.MSE(X, y, theta)
            print(self.MSE(X, y, theta))
            if i > 10 and cost[i] - cost[i-1] < epsilon:
                break
        return theta, cost

