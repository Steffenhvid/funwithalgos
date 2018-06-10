import numpy as np


class LinearClassification:
    pass

    def generate_random_data(self, n):
        '''
        :param n: Size of dataset to return
        :return: A 2d linearly seperable data set.
        '''
        return np.concatenate((np.random.random_integers(0,10, (int(n/2), 2)),np.random.random_integers(11,20, (int(n/2), 2))))