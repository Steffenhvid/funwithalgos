import LinearRegression
from sklearn.datasets import load_boston
import numpy as np

if __name__ == "__main__":
    data = load_boston()
    X_train, y_train = np.insert(data['data'][:int((506*4)/5)],0 ,1, axis=1), data['target'][:int((506*4)/5)]
    X_test, y_test = np.insert(data['data'][int((506*4)/5):], 0, 1, axis=1), data['target'][int((506*4)/5):]
    model = LinearRegression.LinearRegression()
    theta = np.zeros(X_train.shape[1])
    print(model.gradient_descent(X_train, y_train))
