from SupervisedLearning import SupervisedLearning
import numpy as np


class LogisticRegression(SupervisedLearning):
    pass

    def sigmoid(self, X, theta):
        '''
        Computes the sigmoid given the data and the weights
        :param X: numpy array shape (n,d)
        :param theta: numpy array shape (d,1)
        :return: z transformed by the sigmoid function (signal)
        '''
        z = X.dot(theta)
        sig = 1/(1+np.exp(-z))
        assert sig.shape == z.shape
        return sig

    def cost_function(self, X, y, theta):
        '''
        :param X: np.array shape (n,d)
        :param y: np.array shape (n,1)
        :param theta: np.array shape (d,1)
        :param reg: scalar - regularization parameter
        :return:
            cost:
        '''
        z = self.sigmoid(X, theta)
        n = len(z)
        cost = -sum(y * np.log(z) + (1-y) * np.log(1-z))/n
        return cost

