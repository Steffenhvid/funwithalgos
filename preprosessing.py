import numpy as np
from sklearn.datasets import load_boston

data = load_boston()

X, y = data['data'], data['target']

def mean_norm(X):
    '''
    Return the mean normalized dataset, quickens gradient descent
    :param X: numpy.array (n,d)
    :return: numpy.array (n,d)
    '''
    return ((X-X.mean(axis=0))/X.std(axis=0))

def split_data(X, y, split=0.75):
    '''
    Split the data into a training and test set
    :param X: numpy.array (n,d)
    :param y: numpy.array (n,)
    :param split: scalar [0,1], size of the test set
    :return:
        X_train: numpy.array (n*split, d)
        y_train: numpy.array (n*(split))
        X_test: numpy.array (n*(1-split), d)
        y_test: numpy.array (n*(1-split), )
    '''
    X_train = X[:int(len(y)*split),:]
    y_train = y[:int(len(y) * split)]
    X_test = X[int(len(y) * (1-split)):, :]
    y_test = y[int(len(y) * (1-split)):]
    return X_train, y_train, X_test, y_test
