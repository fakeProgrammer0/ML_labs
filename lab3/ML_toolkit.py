'''Some functions used extensively in machine learning labs

'''

import math
import numpy as np


def sign_helper(a, threshold=0, sign_threshold=1):
    ''' A helper method to perform the sign operation on each element of a given array a
    :param a: An ndarray
    :param threshold: The threshold of the sign function, default 0.
                        sign(x) = -1 where x < threshold
                        sign(x) = +1 where x > threshold
                        sign(x) = sign_threshold where x = threshold
    :param sign_threshold: default 1
    :return: The result of the sign operation on the given array a
    '''
    if sign_threshold != 1 and sign_threshold != -1:
        raise ValueError('sign_threshold must be -1 or +1')
    # if not (-1 < threshold < 1):
    #     raise ValueError('threshold must be between -1 and 1')
    sign_a = np.zeros(a.shape)
    sign_a += a > threshold
    sign_a -= a < threshold
    sign_a += sign_threshold * (a == threshold)
    return sign_a

def exp_loss(y_true, y_pred):
    '''calculate the exponential loss of groundtruth labels y_true and predictive labels y_pred

    :param y_true: an ndarray
    :param y_pred: an ndarray
    :return: the exponential loss
    '''
    if y_true.shape != y_pred.shape:
        raise Exception('The shape of y_true must be the same as y_pred')
    return np.exp(-y_true * y_pred).sum() / y_true.size

