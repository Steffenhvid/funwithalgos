import LinearRegression
from preprosessing import *
from sklearn.datasets import load_boston
import numpy as np

if __name__ == "__main__":
    data = load_boston()
    X = mean_norm(data['data']) # Normalizes the data set
    y = data['target']
    X_train, y_train, X_test, y_test = split_data(X,y)
    model = LinearRegression.LinearRegression()
    theta, cost = model.gradient_descent(X_train, y_train)
    print("Final Cost {}".format(model.MSE(X_train, y_train, theta)))
    print("Predicted Cost {}".format(model.MSE(X_test, y_test, theta)))
