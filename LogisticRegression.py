import numpy as np

class LogisticRegression:
    pass

    def sigmoid(self, z):
        '''
        Computes the sigmoid given a the signal
        :param z: numpy array shape (d,)
        :return: z transformed by the sigmoid function
        '''
        sig = 1/(1+np.exp(-z))
        assert sig.shape == z.shape
        return sig

    def cost_function(self, X, y, w, reg=0):
        '''
        Compute the (regularized) cross entropy and the gradient under the logistic regression model
        using data X, targets y, weight vector w (and regularization reg)

        The L2 regularization is 1/2 reg*|w_{1,d}|^2 i.e. w[0], the bias, is not regularized
        :param X: np.array shape (n,d)
        :param y: np.array shape (n,)
        :param w: np.array shape (d,)
        :param reg: scalar - regularization parameter
        :return:
            cost: scalar the cross entropy cost of logistic regression with data X,y using regularization parameter reg
            grad: np.arrray shape(n,d) gradient of cost at w with regularization value reg
        '''
        cost = 0
        # Computing the cost
        z = np.dot(w, X.T)
        logs = self.sigmoid(z)
        ln_log1 = np.log(logs)
        ln_log2 = np.log(1 - logs)
        fp = y * ln_log1
        sp = (1 - y) * ln_log2
        sums = np.sum(fp + sp)
        cost -= (1 / len(y)) * sums
        cost += (1 / 2) * reg * (len(w) - 1) ** 2

        # Computing the gradient
        z2 = np.dot(X, w)
        logs2 = self.sigmoid(z2)
        inner = y - logs2
        prod = np.dot(-X.T, inner)
        grad = (1 / len(y)) * prod
        grad += reg * w * (1 / len(y))
        ### END CODE
        assert grad.shape == w.shape
        return cost, grad

