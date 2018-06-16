import LinearRegression
from preprosessing import *
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression as LR
import numpy as np

if __name__ == "__main__":
    data = load_boston()
    X = mean_norm(data['data']) # Normalizes the data set
    y = data['target']
    X_train, y_train, X_test, y_test = split_data(X,y)
    model = LinearRegression.LinearRegression()
    theta, cost = model.gradient_descent(X_train, y_train)
    print(mean_norm(X_test).dot(theta))

    model1 = LR()
    model1.fit(X_train, y_train)
    print(model1.predict(mean_norm(X_test)))
