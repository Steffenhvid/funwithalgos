import numpy as np

class SupervisedLearning:

    def generate_random_data(self, n):
        '''
        :param n: Size of dataset to return
        :return:
            X: A 2d linearly seperable data set
            y: Label set for X
        '''
        X = np.concatenate((np.random.random_integers(0,10, (int(n/2), 2)),np.random.random_integers(11,20, (int(n/2), 2))))
        y = np.concatenate((np.array([0 for _ in range(int(n/2))]), np.array([1 for _ in range(int(n/2))])))
        assert len(X) == len(y)
        return X, y